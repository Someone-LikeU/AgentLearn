# encoding: utf-8
# @Time    : 2026/04/24
import subprocess
import sys
import time

from mcp_client import MCPClient


def test_subprocess_mode():
    """Test subprocess mode (STDIO communication)"""
    print("=" * 60)
    print("Test Mode: Subprocess Mode (STDIO)")
    print("=" * 60)

    client = MCPClient(mode="subprocess")
    client.start()

    print("\n[1] Ping Test")
    result = client.ping()
    print(f"    Result: {result}")

    print("\n[2] List Tools")
    tools = client.list_tools()
    print(f"    Tool count: {len(tools)}")
    for tool in tools:
        print(f"    - {tool.get('name')}: {tool.get('description', '')[:50]}...")

    print("\n[3] Call query_weather")
    weather = client.call_tool("query_weather", {"city": "Beijing", "days": 3})
    print(f"    Result: {weather}")

    print("\n[4] Call query_flight_tickets")
    flights = client.call_tool("query_flight_tickets", {
        "from_city": "北京",
        "to_city": "上海",
        "direct": False,
    })
    print(f"    Result: {flights}")

    client.close()
    print("\nSubprocess mode test PASSED")


def test_tcp_mode():
    """Test TCP mode"""
    print("\n" + "=" * 60)
    print("Test Mode: TCP Mode")
    print("=" * 60)

    server_process = None
    try:
        print("\n[0] Start TCP server...")
        server_process = subprocess.Popen(
            [sys.executable, "mcp_server.py", "--mode", "tcp"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
        )
        time.sleep(1)
        print("    Server started (port 7777)")

        print("\n[1] Connect to TCP server")
        client = MCPClient(mode="tcp", host="127.0.0.1", port=7777)
        client.start()
        print("    Connected")

        print("\n[2] Ping Test")
        result = client.ping()
        print(f"    Result: {result}")

        print("\n[3] List Tools")
        tools = client.list_tools()
        print(f"    Tool count: {len(tools)}")
        for tool in tools:
            print(f"    - {tool.get('name')}: {tool.get('description', '')[:50]}...")

        print("\n[4] Call query_weather")
        weather = client.call_tool("query_weather", {"city": "Shanghai", "days": 5})
        print(f"    Result: {weather}")

        print("\n[5] Call query_flight_tickets")
        flights = client.call_tool("query_flight_tickets", {
            "from_city": "北京",
            "to_city": "贵阳",
            "direct": True,
        })
        print(f"    Result: {flights}")

        client.close()
        print("\nTCP mode test PASSED")

    finally:
        if server_process:
            server_process.terminate()
            server_process.wait(timeout=3)
            print("\nServer stopped")


if __name__ == "__main__":
    print("MCP Client Test")
    print("Verify subprocess mode and TCP mode work correctly\n")

    print("[Part 1] Test Subprocess Mode")
    test_subprocess_mode()

    print("\n" + "=" * 60)
    print("[Part 2] Test TCP Mode")
    print("=" * 60)

    test_tcp_mode()

    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
