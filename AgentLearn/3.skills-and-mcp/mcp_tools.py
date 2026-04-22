# encoding: utf-8
# @Time    : 2026/04/22 00:00
import datetime
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any


@dataclass
class MCPTool:
	name: str
	description: str
	parameters: dict[str, Any]
	handler: Any


class MCPToolsRegistry:
	def __init__(self):
		self._station_code_cache: dict[str, str] | None = None
		self._city_code_cache: dict[str, str] = {
			"北京": "BJS",
			"上海": "SHA",
			"广州": "CAN",
			"深圳": "SZX",
			"成都": "CTU",
			"杭州": "HGH",
		}
		self._tools: dict[str, MCPTool] = {}
		self._register_tools()

	def _http_get_json(self, url: str, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None):
		query = ""
		if params:
			query = urllib.parse.urlencode(params)
		full_url = f"{url}?{query}" if query else url
		request = urllib.request.Request(full_url, headers=headers or {})
		with urllib.request.urlopen(request, timeout=20) as response:
			body = response.read().decode("utf-8")
			return json.loads(body)

	def _http_get_text(self, url: str, params: dict[str, Any] | None = None):
		query = ""
		if params:
			query = urllib.parse.urlencode(params)
		full_url = f"{url}?{query}" if query else url
		with urllib.request.urlopen(full_url, timeout=20) as response:
			return response.read().decode("utf-8")

	def _register_tools(self):
		self._tools["query_weather"] = MCPTool(
			name="query_weather",
			description="查询某个城市未来天气，days默认15天，最大16天。",
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
		self._tools["query_train_tickets"] = MCPTool(
			name="query_train_tickets",
			description="调用12306查询火车票余票信息。",
			parameters={
				"type": "object",
				"properties": {
					"from_city": {"type": "string", "description": "出发城市，如北京"},
					"to_city": {"type": "string", "description": "到达城市，如上海"},
					"date": {"type": "string", "description": "出发日期，格式YYYY-MM-DD"},
				},
				"required": ["from_city", "to_city", "date"],
			},
			handler=self.query_train_tickets,
		)
		self._tools["query_flight_tickets"] = MCPTool(
			name="query_flight_tickets",
			description="调用携程公开接口查询机票价格趋势（最低价接口）。",
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
		return [
			{"name": tool.name, "description": tool.description, "parameters": tool.parameters}
			for tool in self._tools.values()
		]

	def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
		tool = self._tools.get(name)
		if tool is None:
			raise ValueError(f"Unknown MCP tool '{name}'")
		return tool.handler(**arguments)

	def query_weather(self, city: str, days: int = 15) -> dict[str, Any]:
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

	def _load_station_codes(self) -> dict[str, str]:
		if self._station_code_cache is not None:
			return self._station_code_cache
		text = self._http_get_text("https://kyfw.12306.cn/otn/resources/js/framework/station_name.js")
		entries = re.findall(r"@([^|]+)\|([^|]+)\|([^|]+)\|([^|]+)\|", text)
		mapping: dict[str, str] = {}
		for pinyin, zh_name, code, _ in entries:
			mapping[zh_name] = code
			mapping[pinyin] = code
			mapping[code] = code
		self._station_code_cache = mapping
		return mapping

	def query_train_tickets(self, from_city: str, to_city: str, date: str) -> dict[str, Any]:
		datetime.datetime.strptime(date, "%Y-%m-%d")
		stations = self._load_station_codes()
		from_station = stations.get(from_city)
		to_station = stations.get(to_city)
		if not from_station or not to_station:
			return {
				"error": "无法识别城市/车站名称，请尝试中文站名（如北京、上海）",
				"from_city": from_city,
				"to_city": to_city,
			}
		payload = self._http_get_json(
			"https://kyfw.12306.cn/otn/leftTicket/query",
			{
				"leftTicketDTO.train_date": date,
				"leftTicketDTO.from_station": from_station,
				"leftTicketDTO.to_station": to_station,
				"purpose_codes": "ADULT",
			},
		)
		results = payload.get("data", {}).get("result", [])
		tickets = []
		for row in results[:20]:
			parts = row.split("|")
			if len(parts) < 33:
				continue
			tickets.append(
				{
					"train_no": parts[3],
					"from_station_code": parts[6],
					"to_station_code": parts[7],
					"depart_time": parts[8],
					"arrive_time": parts[9],
					"duration": parts[10],
					"business_seat": parts[32],
					"first_class": parts[31],
					"second_class": parts[30],
					"soft_sleep": parts[23],
					"hard_sleep": parts[28],
					"hard_seat": parts[29],
					"no_seat": parts[26],
				}
			)
		return {
			"date": date,
			"from_city": from_city,
			"to_city": to_city,
			"count": len(tickets),
			"tickets": tickets,
		}

	def _resolve_ctrip_city_code(self, city: str) -> str | None:
		if city in self._city_code_cache:
			return self._city_code_cache[city]
		payload = self._http_get_json(
			"https://flights.ctrip.com/international/search/api/poi/getPoiSuggest",
			{"keyword": city},
		)
		for item in payload.get("data") or []:
			city_code = item.get("cityCode") or item.get("iataCode")
			if city_code:
				self._city_code_cache[city] = city_code
				return city_code
		return None

	def query_flight_tickets(self, from_city: str, to_city: str, direct: bool = False) -> dict[str, Any]:
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
