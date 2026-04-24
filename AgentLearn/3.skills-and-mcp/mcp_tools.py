# encoding: utf-8
# @Time    : 2026/04/22
import json
import ssl
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# 禁用 SSL 证书验证
ssl._create_default_https_context = ssl._create_unverified_context


@dataclass
class MCPTool:
	"""
	封装一个MCP工具类
	"""
	name: str
	description: str
	parameters: dict[str, Any]
	handler: Any


class MCPToolsRegistry:
	"""
	MCP 工具注册、管理中心
	"""
	def __init__(self):
		# 城市编码缓存
		self._city_code_cache: dict[str, str] = {
			"北京": "BJS",
			"上海": "SHA",
			"广州": "CAN",
			"深圳": "SZX",
			"成都": "CTU",
			"杭州": "HGH",
			"贵阳": "KWE"
		}
		self._city_code_cache_file = Path(__file__).resolve().parent / "cache" / "city_code_cache.json"
		self._city_code_cache: dict[str, str] = self._load_city_code_cache()
		# 工具列表
		self._tools: dict[str, MCPTool] = {}
		# 注册工具
		self._register_tools()

	def _load_city_code_cache(self) -> dict[str, str]:
		"""从本地JSON文件加载城市三字码缓存。"""
		try:
			with open(self._city_code_cache_file, "r", encoding="utf-8") as f:
				data = json.load(f)
			if isinstance(data, dict):
				# 仅保留字符串键值，避免脏数据污染
				return {str(k): str(v).upper() for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
		except FileNotFoundError:
			return {}
		except json.JSONDecodeError:
			return {}
		return {}

	def _http_get_json(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None):
		"""
		http请求得到json响应
		:param url: 地址
		:param params: 参数
		:param headers: headers
		:return:
		"""
		query = ""
		if params:
			query = urllib.parse.urlencode(params)
		full_url = f"{url}?{query}" if query else url
		request = urllib.request.Request(full_url, headers=headers or {})
		with urllib.request.urlopen(request, timeout=20) as response:
			# 使用"utf-8-sig"解码解决处理不了BOM的问题
			body = response.read().decode("utf-8-sig")
			if not body:
				raise ValueError(f"Empty response from {url}")
			return json.loads(body)

	def _http_get_text(self, url: str, params: dict[str, Any] | None = None):
		"""
		http请求得到文本响应
		:param url: 地址
		:param params: 参数
		:return:
		"""
		query = ""
		if params:
			query = urllib.parse.urlencode(params)
		full_url = f"{url}?{query}" if query else url
		with urllib.request.urlopen(full_url, timeout=20) as response:
			return response.read().decode("utf-8-sig")

	def _register_tools(self):
		"""
		注册可用的工具
		:return:
		"""
		# 查询某城市天气
		self._tools["query_weather"] = MCPTool(
			name="query_weather",
			description="当用户任务可能涉及到需要查询天气时，调用该工具，例如'帮我规划未来3天假期去北京的旅游行程'、'明天需不需要穿羽绒服'等。该工具查询某个城市的未来天气，days参数默认15天，最大16天。",
			parameters={
				"type": "object",
				"properties": {
					"city": {"type": "string", "description": "城市名称，如北京、上海"},
					"days": {"type": "integer", "description": "未来天数，默认15，最大16"},
				},
				"required": ["city"],
			},
			handler=self.query_weather,
		)
		# 机票查询工具
		self._tools["query_flight_tickets"] = MCPTool(
			name="query_flight_tickets",
			description="该工具调用携程公开接口查询机票价格趋势（最低价接口）。当用户任务可能涉及到查询机票信息时，调用该工具，例如'帮我规划未来5天假期去北京的旅游行程'、'未来10天有没有去北京的低于800的机票？'等。",
			parameters={
				"type": "object",
				"properties": {
					"from_city": {"type": "string", "description": "出发城市，如北京"},
					"to_city": {"type": "string", "description": "到达城市，如上海"},
					"direct": {"type": "boolean", "description": "是否直飞，默认false"},
				},
				"required": ["from_city", "to_city"],
			},
			handler=self.query_flight_tickets,
		)

	def list_tools(self) -> list[dict[str, Any]]:
		"""
		返回可用工具列表，按照OpenAI的工具调用格式
		:return:
		"""
		return [
			{"name": tool.name, "description": tool.description, "parameters": tool.parameters}
			for tool in self._tools.values()
		]

	def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
		"""
		工具调用
		:param name: 工具名
		:param arguments: 参数列表
		:return:
		"""
		tool = self._tools.get(name)
		if tool is None:
			raise ValueError(f"Unknown MCP tool '{name}'")
		return tool.handler(**arguments)

	def query_weather(self, city: str, days: int = 15) -> dict[str, Any]:
		"""
		调API查询天气
		:param city: 	城市名
		:param days: 	天数
		:return:
		"""
		days = max(1, min(days, 16))
		geo_data = self._http_get_json(
			"https://geocoding-api.open-meteo.com/v1/search",
			{"name": city, "count": 1, "language": "zh", "format": "json"},
		)
		results = geo_data.get("results") or []
		if not results:
			return {"city": city, "error": "未找到该城市的地理信息"}
		location = results[0]
		forecast_data = self._http_get_json(
			"https://api.open-meteo.com/v1/forecast",
			{
				"latitude": location["latitude"],
				"longitude": location["longitude"],
				"daily": "weathercode,temperature_2m_max,temperature_2m_min,precipitation_probability_max",
				"forecast_days": days,
				"timezone": "Asia/Shanghai",
			},
		)
		payload = forecast_data.get("daily", {})
		times = payload.get("time", [])
		max_temps = payload.get("temperature_2m_max", [])
		min_temps = payload.get("temperature_2m_min", [])
		rain_prob = payload.get("precipitation_probability_max", [])
		codes = payload.get("weathercode", [])
		forecast = []
		for idx, day in enumerate(times):
			forecast.append(
				{
					"date": day,
					"max_temp": max_temps[idx] if idx < len(max_temps) else None,
					"min_temp": min_temps[idx] if idx < len(min_temps) else None,
					"rain_probability": rain_prob[idx] if idx < len(rain_prob) else None,
					"weather_code": codes[idx] if idx < len(codes) else None,
				}
			)
		return {
			"city": location.get("name", city),
			"country": location.get("country"),
			"timezone": "Asia/Shanghai",
			"days": days,
			"forecast": forecast,
		}

	def _resolve_ctrip_city_code(self, city: str) -> str | None:
		"""
		解析携程城市三字码。
		响应结构兼容以下两类：
		1) data 为列表；
		2) data 为嵌套字典（热门是列表，其它分组按首字母再嵌套列表）。
		"""
		# 在缓存里就直接返回
		if city in self._city_code_cache:
			return self._city_code_cache[city]

		payload = self._http_get_json(
			"https://flights.ctrip.com/itinerary/api/poi/get",
			{"query": city},
			headers={"Referer": "https://flights.ctrip.com/"},
		)

		def _collect_items(node):
			if isinstance(node, list):
				for item in node:
					if isinstance(item, dict):
						yield item
			elif isinstance(node, dict):
				for value in node.values():
					yield from _collect_items(value)

		def _extract_code(raw_data: str) -> str | None:
			# 示例: "Fuzhou|福州(FOC)|258|FOC"
			parts = raw_data.split("|")
			if len(parts) >= 4 and parts[3]:
				return parts[3].upper()
			if len(parts) >= 2 and "(" in parts[1] and ")" in parts[1]:
				return parts[1].split("(")[-1].split(")")[0].upper()
			return None

		items = list(_collect_items(payload.get("data")))
		if not items:
			return None

		# 优先精确匹配 display（例如 city=广州，命中 display=广州）
		for item in items:
			if str(item.get("display", "")).strip() == city:
				code = _extract_code(str(item.get("data", "")))
				if code:
					self._city_code_cache[city] = code
					return code

		# 次选：匹配 data 中中文城市名字段（如 "福州(FOC)"）
		for item in items:
			raw_data = str(item.get("data", ""))
			parts = raw_data.split("|")
			if len(parts) >= 2 and city in parts[1]:
				code = _extract_code(raw_data)
				if code:
					self._city_code_cache[city] = code
					return code

		# 最后兜底：取第一个可解析条目，避免完全失败
		for item in items:
			code = _extract_code(str(item.get("data", "")))
			if code:
				self._city_code_cache[city] = code
				return code
		return None

	def query_flight_tickets(self, from_city: str, to_city: str, direct: bool = False) -> dict[str, Any]:
		"""
		查询航班信息
		:param from_city: 出发城市
		:param to_city: 到达城市
		:param direct: 是否直飞，默认False
		:return:
		"""
		from_code = self._resolve_ctrip_city_code(from_city)
		to_code = self._resolve_ctrip_city_code(to_city)
		if not from_code or not to_code:
			return {
				"error": "携程城市编码解析失败，请尝试常见城市中文名",
				"from_city": from_city,
				"to_city": to_city,
			}
		data = self._http_get_json(
			"https://flights.ctrip.com/itinerary/api/12808/lowestPrice",
			{
				"flightWay": "Oneway",
				"dcity": from_code,
				"acity": to_code,
				"direct": str(bool(direct)).lower(),
			},
			headers={"Referer": "https://flights.ctrip.com/"},
		)
		return {
			"from_city": from_city,
			"to_city": to_city,
			"from_code": from_code,
			"to_code": to_code,
			"direct": direct,
			"raw": data,
		}


if __name__ == "__main__":
	registry = MCPToolsRegistry()
	print(json.dumps(registry.list_tools(), ensure_ascii=False, indent=2))
