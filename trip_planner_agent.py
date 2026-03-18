"""多智能体旅行规划系统 """

import json
import requests
from typing import Dict, Any, List, Annotated, TypedDict, Optional, Literal, Callable
from datetime import datetime, timedelta

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages  # 如果需要消息累加可启用
from pydantic import BaseModel, Field

from ..services.llm_service import get_llm
from ..models.schemas import TripRequest, TripPlan, DayPlan, Attraction, Meal, WeatherInfo, Location, Hotel
from ..config import get_settings


# ============ LangChain 工具定义（替换原来的 MCPTool）============
@tool
def amap_maps_text_search(keywords: str, city: str) -> str:
    """高德地图文本搜索（景点/酒店通用）"""
    settings = get_settings()
    url = "https://restapi.amap.com/v3/place/text"
    params = {
        "key": settings.amap_api_key,
        "keywords": keywords,
        "city": city,
        "output": "json",
        "extensions": "all",
    }
    try:
        resp = requests.get(url, params=params, timeout=10).json()
        if resp.get("status") != "1":
            return f"搜索失败: {resp.get('info')}"
        
        pois = resp.get("pois", [])[:8]  # 限制返回数量
        summary = f"✅ 在 {city} 搜索 “{keywords}” 找到 {len(pois)} 个结果：\n"
        for p in pois:
            loc = p.get("location", "0,0").split(",")
            summary += (
                f"• {p.get('name')} | "
                f"地址: {p.get('address')} | "
                f"坐标: (lng:{loc[0]}, lat:{loc[1]}) | "
                f"类型: {p.get('type')}\n"
            )
        return summary
    except Exception as e:
        return f"工具调用异常: {str(e)}"


@tool
def amap_maps_weather(city: str) -> str:
    """高德天气查询（支持多天预报）"""
    settings = get_settings()
    url = "https://restapi.amap.com/v3/weather/weatherInfo"
    params = {
        "key": settings.amap_api_key,
        "city": city,
        "extensions": "all",
    }
    try:
        resp = requests.get(url, params=params, timeout=10).json()
        if resp.get("status") != "1":
            return f"天气查询失败: {resp.get('info')}"
        
        forecasts = resp.get("forecasts", [{}])[0].get("casts", [])
        summary = f"🌤️ {city} 天气预报（未来几天）：\n"
        for day in forecasts:
            summary += (
                f"日期: {day['date']} | "
                f"白天: {day['dayweather']} {day['daytemp']}°C | "
                f"夜晚: {day['nightweather']} {day['nighttemp']}°C | "
                f"风: {day['daywind']} {day['daypower']}级\n"
            )
        return summary
    except Exception as e:
        return f"工具调用异常: {str(e)}"


# ============ 更新后的 Agent 提示词（去掉强制 [TOOL_CALL] 格式）============
ATTRACTION_AGENT_PROMPT = """你是景点搜索专家。
你的任务是根据用户提供的城市和偏好，使用工具搜索合适的景点。
必须使用 amap_maps_text_search 工具，不要自己编造任何景点信息。
返回搜索结果的总结即可。"""

WEATHER_AGENT_PROMPT = """你是天气查询专家。
你的任务是查询指定城市的天气信息。
必须使用 amap_maps_weather 工具，不要自己编造天气数据。
返回清晰的天气预报总结。"""

HOTEL_AGENT_PROMPT = """你是酒店推荐专家。
你的任务是根据城市搜索合适的酒店。
必须使用 amap_maps_text_search 工具，并将 keywords 设置为“酒店”或“宾馆”。
返回酒店搜索结果总结。"""

PLANNER_AGENT_PROMPT = """你是行程规划专家。你的任务是根据景点信息和天气信息,生成详细的旅行计划。

请严格按照以下JSON格式返回旅行计划:
```json
{
  "city": "城市名称",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "days": [
    {
      "date": "YYYY-MM-DD",
      "day_index": 0,
      "description": "第1天行程概述",
      "transportation": "交通方式",
      "accommodation": "住宿类型",
      "hotel": {
        "name": "酒店名称",
        "address": "酒店地址",
        "location": {"longitude": 116.397128, "latitude": 39.916527},
        "price_range": "300-500元",
        "rating": "4.5",
        "distance": "距离景点2公里",
        "type": "经济型酒店",
        "estimated_cost": 400
      },
      "attractions": [
        {
          "name": "景点名称",
          "address": "详细地址",
          "location": {"longitude": 116.397128, "latitude": 39.916527},
          "visit_duration": 120,
          "description": "景点详细描述",
          "category": "景点类别",
          "ticket_price": 60
        }
      ],
      "meals": [
        {"type": "breakfast", "name": "早餐推荐", "description": "早餐描述", "estimated_cost": 30},
        {"type": "lunch", "name": "午餐推荐", "description": "午餐描述", "estimated_cost": 50},
        {"type": "dinner", "name": "晚餐推荐", "description": "晚餐描述", "estimated_cost": 80}
      ]
    }
  ],
  "weather_info": [
    {
      "date": "YYYY-MM-DD",
      "day_weather": "晴",
      "night_weather": "多云",
      "day_temp": 25,
      "night_temp": 15,
      "wind_direction": "南风",
      "wind_power": "1-3级"
    }
  ],
  "overall_suggestions": "总体建议",
  "budget": {
    "total_attractions": 180,
    "total_hotels": 1200,
    "total_meals": 480,
    "total_transportation": 200,
    "total": 2060
  }
}
```

**重要提示:**
1. weather_info数组必须包含每一天的天气信息
2. 温度必须是纯数字(不要带°C等单位)
3. 每天安排2-3个景点
4. 考虑景点之间的距离和游览时间
5. 每天必须包含早中晚三餐
6. 提供实用的旅行建议
7. **必须包含预算信息**:
   - 景点门票价格(ticket_price)
   - 餐饮预估费用(estimated_cost)
   - 酒店预估费用(estimated_cost)
   - 预算汇总(budget)包含各项总费用
"""


class AgentEnvelope(BaseModel):
    sender: str
    kind: Literal["TASK", "RESULT", "ERROR"]
    task: str
    recipient: Optional[str] = None
    payload: Dict[str, Any] = Field(default_factory=dict)
    ts: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


def _append_envelopes(left: Optional[List[Dict[str, Any]]], right: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    return (left or []) + (right or [])


def _mk_msg(sender: str, kind: Literal["TASK", "RESULT", "ERROR"], task: str, payload: Dict[str, Any], recipient: Optional[str] = None) -> Dict[str, Any]:
    return AgentEnvelope(sender=sender, kind=kind, task=task, payload=payload, recipient=recipient).model_dump()


def _latest_msg(messages: List[Dict[str, Any]], *, kind: Optional[str] = None, task: Optional[str] = None, recipient: Optional[str] = None) -> Optional[Dict[str, Any]]:
    for m in reversed(messages):
        if kind and m.get("kind") != kind:
            continue
        if task and m.get("task") != task:
            continue
        if recipient and m.get("recipient") != recipient:
            continue
        return m
    return None



# ============ Graph State 定义 ============
class TripState(TypedDict):
    messages: Annotated[List[Dict[str, Any]], _append_envelopes]
    request: TripRequest
    attraction_info: str
    weather_info: str
    hotel_info: str
    final_plan: TripPlan | None


# ============ 多智能体旅行规划系统（LangGraph 版）============
class MultiAgentTripPlanner:
    """多智能体旅行规划系统 - 使用 LangChain v1 + LangGraph v1"""

    def __init__(self):
        print("🔄 开始初始化 LangGraph 多智能体旅行规划系统...")

        try:
            self.llm = get_llm()
            settings = get_settings()

            print(f"  - 使用模型: {self.llm.model_name}")
            self.planner_llm = self.llm

            # 创建 ReAct Agent（使用 2026 新标准 create_agent）
            print("  - 创建景点搜索 Agent...")
            self.attraction_agent = create_agent(
                model=self.llm,
                tools=[amap_maps_text_search],
                system_prompt=ATTRACTION_AGENT_PROMPT,  # create_agent 使用 system_prompt
                name="attraction_agent",
            )

            print("  - 创建天气查询 Agent...")
            self.weather_agent = create_agent(
                model=self.llm,
                tools=[amap_maps_weather],
                system_prompt=WEATHER_AGENT_PROMPT,
                name="weather_agent",
            )

            print("  - 创建酒店推荐 Agent...")
            self.hotel_agent = create_agent(
                model=self.llm,
                tools=[amap_maps_text_search],
                system_prompt=HOTEL_AGENT_PROMPT,
                name="hotel_agent",
            )

            print("  - 创建行程规划Agent...")
            self.planner_agent = create_agent(
                model=self.llm,
                system_prompt=PLANNER_AGENT_PROMPT,
                name="planner_agent",
            )

            # 构建 StateGraph
            print("  - 构建 LangGraph 工作流...")
            self.graph = self._build_graph()

            print("✅ LangGraph 多智能体系统初始化成功")

        except Exception as e:
            print(f"❌ 初始化失败: {str(e)}")
            import traceback
            traceback.print_exc()
            raise

    def _build_graph(self):
        """构建 StateGraph（核心协调 Agent + 专业化子 Agent + 结果聚合）"""
        graph = StateGraph(TripState)

        # 定义节点
        graph.add_node("coordinator", self._coordinator_node)
        graph.add_node("attraction", self._attraction_node)
        graph.add_node("weather", self._weather_node)
        graph.add_node("hotel", self._hotel_node)
        graph.add_node("planner", self._planner_node)

        graph.add_edge(START, "coordinator")

        # 并行执行（景点、天气、酒店搜索可以同时进行）
        graph.add_edge("coordinator", "attraction")
        graph.add_edge("coordinator", "weather")
        graph.add_edge("coordinator", "hotel")
        
        # 汇聚到行程规划器
        graph.add_edge("attraction", "planner")
        graph.add_edge("weather", "planner")
        graph.add_edge("hotel", "planner")
        
        graph.add_edge("planner", END)

        return graph.compile()

    # ============== 各节点函数 ==============
    def _coordinator_node(self, state: TripState) -> TripState:
        req = state["request"]
        print(f"🧩 [协调器] 开始任务拆分与调度: {req.city} {req.start_date}-{req.end_date}")

        keywords = req.preferences[0] if req.preferences else "景点"
        tasks = [
            _mk_msg(
                sender="coordinator",
                kind="TASK",
                task="attraction_search",
                recipient="attraction_agent",
                payload={"city": req.city, "keywords": keywords, "query": f"请搜索 {req.city} 的 {keywords} 相关景点。"},
            ),
            _mk_msg(
                sender="coordinator",
                kind="TASK",
                task="weather_query",
                recipient="weather_agent",
                payload={"city": req.city, "query": f"请查询 {req.city} 的天气信息（覆盖旅行期间）。"},
            ),
            _mk_msg(
                sender="coordinator",
                kind="TASK",
                task="hotel_search",
                recipient="hotel_agent",
                payload={"city": req.city, "accommodation": req.accommodation, "query": f"请搜索 {req.city} 的 {req.accommodation} 酒店（推荐经济型或中档）。"},
            ),
        ]
        return {"messages": tasks}

    def _attraction_node(self, state: TripState) -> TripState:
        req = state["request"]
        print(f"🔍 [景点搜索] 正在为 {req.city} 寻找灵感...")
        task = _latest_msg(state.get("messages", []), kind="TASK", task="attraction_search", recipient="attraction_agent")
        query = task["payload"]["query"] if task and task.get("payload") else f"请搜索 {req.city} 的 景点 相关景点。"

        response = self.attraction_agent.invoke({"messages": [HumanMessage(content=query)]})
        print(f"✅ [景点搜索] 完成！已获取景点信息。")
        content = response["messages"][-1].content
        return {
            "attraction_info": content,
            "messages": [
                _mk_msg(sender="attraction_agent", kind="RESULT", task="attraction_search", recipient="coordinator", payload={"text": content})
            ],
        }

    def _weather_node(self, state: TripState) -> TripState:
        req = state["request"]
        print(f"🌤️ [天气查询] 正在获取 {req.city} 的最新天气预报...")
        task = _latest_msg(state.get("messages", []), kind="TASK", task="weather_query", recipient="weather_agent")
        query = task["payload"]["query"] if task and task.get("payload") else f"请查询 {req.city} 的天气信息（覆盖旅行期间）。"

        response = self.weather_agent.invoke({"messages": [HumanMessage(content=query)]})
        print(f"✅ [天气查询] 完成！已获取天气数据。")
        content = response["messages"][-1].content
        return {
            "weather_info": content,
            "messages": [
                _mk_msg(sender="weather_agent", kind="RESULT", task="weather_query", recipient="coordinator", payload={"text": content})
            ],
        }

    def _hotel_node(self, state: TripState) -> TripState:
        req = state["request"]
        print(f"🏨 [酒店推荐] 正在搜索 {req.city} 的合适住处...")
        task = _latest_msg(state.get("messages", []), kind="TASK", task="hotel_search", recipient="hotel_agent")
        query = task["payload"]["query"] if task and task.get("payload") else f"请搜索 {req.city} 的 {req.accommodation} 酒店（推荐经济型或中档）。"

        response = self.hotel_agent.invoke({"messages": [HumanMessage(content=query)]})
        print(f"✅ [酒店推荐] 完成！已获取酒店推荐。")
        content = response["messages"][-1].content
        return {
            "hotel_info": content,
            "messages": [
                _mk_msg(sender="hotel_agent", kind="RESULT", task="hotel_search", recipient="coordinator", payload={"text": content})
            ],
        }

    def _planner_node(self, state: TripState) -> TripState:
        print(f"📝 [行程规划] 正在整合所有信息生成最终计划...")
        req = state["request"]
        
        attraction_text = state.get("attraction_info", "")
        weather_text = state.get("weather_info", "")
        hotel_text = state.get("hotel_info", "")

        attraction_msg = _latest_msg(state.get("messages", []), kind="RESULT", task="attraction_search")
        weather_msg = _latest_msg(state.get("messages", []), kind="RESULT", task="weather_query")
        hotel_msg = _latest_msg(state.get("messages", []), kind="RESULT", task="hotel_search")

        if attraction_msg and attraction_msg.get("payload", {}).get("text"):
            attraction_text = attraction_msg["payload"]["text"]
        if weather_msg and weather_msg.get("payload", {}).get("text"):
            weather_text = weather_msg["payload"]["text"]
        if hotel_msg and hotel_msg.get("payload", {}).get("text"):
            hotel_text = hotel_msg["payload"]["text"]

        # 优化提示词，确保模型输出 JSON
        context = f"""
基本信息：
- 城市: {req.city}
- 日期: {req.start_date} 至 {req.end_date}（{req.travel_days}天）
- 交通/住宿偏好: {req.transportation} / {req.accommodation}
- 偏好: {', '.join(req.preferences) if req.preferences else '无'}

[景点参考]
{attraction_text}

[天气参考]
{weather_text}

[酒店参考]
{hotel_text}

重要提示：请务必只输出一个符合 TripPlan 结构的合法 JSON 字符串。不要输出任何解释性文字。
"""
        try:
            response = self.planner_agent.invoke({"messages": [HumanMessage(content=context)]})
            content = response["messages"][-1].content

            # 3. 手动解析 JSON
            print("  - 正在解析 LLM 返回的 JSON 内容...")
            json_str = str(content).strip()
            
            # 清洗 Markdown 代码块
            if "```json" in json_str:
                json_str = json_str.split("```json")[1].split("```")[0].strip()
            elif "```" in json_str:
                # 寻找第一个代码块
                parts = json_str.split("```")
                if len(parts) >= 3:
                    json_str = parts[1].strip()
            
            # 尝试定位第一个 { 和最后一个 }
            if "{" in json_str and "}" in json_str:
                start = json_str.find("{")
                end = json_str.rfind("}") + 1
                json_str = json_str[start:end]

            # 在加载前对字段名进行模糊兼容修复（处理 common LLM errors）
            import json
            raw_data = json.loads(json_str)
            
            # 常见字段名纠错
            field_map = {
                "plan": "days",
                "itinerary": "days",
                "daily_plans": "days",
                "suggestions": "overall_suggestions",
                "suggestion": "overall_suggestions",
                "transportation_preference": "transportation",
                "accommodation_preference": "accommodation"
            }
            for old_key, new_key in field_map.items():
                if old_key in raw_data and new_key not in raw_data:
                    print(f"    - 修复字段名: {old_key} -> {new_key}")
                    raw_data[new_key] = raw_data.pop(old_key)
            
            # 再次验证关键字段
            if "days" not in raw_data:
                 print("    ⚠️ 警告: JSON 中缺少 'days' 字段")

            plan = TripPlan.model_validate(raw_data)
            print(f"✅ [行程规划] 完成！最终旅行计划已生成。")
            return {
                "final_plan": plan,
                "messages": [
                    _mk_msg(sender="planner_agent", kind="RESULT", task="trip_plan", recipient="coordinator", payload={"plan": plan.model_dump()})
                ],
            }
            
        except Exception as e:
            print(f"❌ [行程规划] 最终失败: {str(e)}")
            # 记录失败时的部分输出用于调试
            if 'content' in locals():
                print(f"   LLM 输出内容预览: {str(content)[:300]}...")
            raise e

    def plan_trip(self, request: TripRequest) -> TripPlan:
        """主入口：执行 LangGraph 工作流"""
        try:
            print(f"\n{'='*60}")
            print(f"🚀 开始 LangGraph 多智能体协作规划旅行...")
            print(f"目的地: {request.city} | 天数: {request.travel_days}")
            print(f"{'='*60}\n")

            initial_state: TripState = {"request": request, "messages": []}
            result = self.graph.invoke(initial_state)

            print(f"✅ 旅行计划生成完成！")
            return result["final_plan"]

        except Exception as e:
            print(f"❌ LangGraph 执行失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return self._create_fallback_plan(request)

    def _create_fallback_plan(self, request: TripRequest) -> TripPlan:
        """备用计划（与原版保持一致）"""
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        days = []
        for i in range(request.travel_days):
            current = start_date + timedelta(days=i)
            day_plan = DayPlan(
                date=current.strftime("%Y-%m-%d"),
                day_index=i,
                description=f"第{i+1}天行程",
                transportation=request.transportation,
                accommodation=request.accommodation,
                attractions=[
                    Attraction(
                        name=f"{request.city}景点{j+1}",
                        address=f"{request.city}市中心",
                        location=Location(longitude=116.4 + i*0.01, latitude=39.9 + i*0.01),
                        visit_duration=120,
                        description="默认景点（后备方案）",
                        category="景点"
                    )
                    for j in range(2)
                ],
                meals=[
                    Meal(type="breakfast", name=f"早餐", description="当地特色早餐", estimated_cost=30),
                    Meal(type="lunch", name=f"午餐", description="午餐推荐", estimated_cost=50),
                    Meal(type="dinner", name=f"晚餐", description="晚餐推荐", estimated_cost=80),
                ]
            )
            days.append(day_plan)

        return TripPlan(
            city=request.city,
            start_date=request.start_date,
            end_date=request.end_date,
            days=days,
            weather_info=[],
            overall_suggestions="这是后备方案（Agent 执行异常时使用），建议实际使用时检查高德地图最新信息。",
        )


# ============== 单例模式 ==============
_multi_agent_planner = None


def get_trip_planner_agent() -> MultiAgentTripPlanner:
    """获取单例实例"""
    global _multi_agent_planner
    if _multi_agent_planner is None:
        _multi_agent_planner = MultiAgentTripPlanner()
    return _multi_agent_planner
