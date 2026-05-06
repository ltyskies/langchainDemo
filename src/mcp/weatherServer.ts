/**
 * ============================================
 * MCP 天气服务器 (Weather MCP Server)
 * ============================================
 *
 * 这是一个基于 Model Context Protocol (MCP) 的天气查询服务器
 * 它通过 stdio 传输方式暴露天气查询工具
 *
 * 功能：
 *   - 查询指定城市的当前天气
 *   - 查询未来几天的天气预报
 *   - 查询空气质量指数
 *
 * 注意：这是一个模拟服务器，返回的是模拟数据
 * 在实际应用中，可以替换为真实的天气 API 调用
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

/**
 * 模拟天气数据库
 * 包含中国主要城市的天气数据
 */
const weatherDatabase: Record<
  string,
  {
    temperature: number;
    condition: string;
    humidity: number;
    windSpeed: number;
    aqi: number; // 空气质量指数
    forecast: Array<{
      date: string;
      temperature: string;
      condition: string;
    }>;
  }
> = {
  北京: {
    temperature: 25,
    condition: "晴朗",
    humidity: 45,
    windSpeed: 12,
    aqi: 85,
    forecast: [
      { date: "明天", temperature: "23°C ~ 28°C", condition: "多云" },
      { date: "后天", temperature: "22°C ~ 27°C", condition: "小雨" },
      { date: "大后天", temperature: "21°C ~ 26°C", condition: "晴朗" },
    ],
  },
  上海: {
    temperature: 28,
    condition: "多云",
    humidity: 65,
    windSpeed: 15,
    aqi: 72,
    forecast: [
      { date: "明天", temperature: "26°C ~ 30°C", condition: "雷阵雨" },
      { date: "后天", temperature: "25°C ~ 29°C", condition: "小雨" },
      { date: "大后天", temperature: "26°C ~ 31°C", condition: "多云" },
    ],
  },
  广州: {
    temperature: 32,
    condition: "雷阵雨",
    humidity: 80,
    windSpeed: 8,
    aqi: 55,
    forecast: [
      { date: "明天", temperature: "29°C ~ 33°C", condition: "雷阵雨" },
      { date: "后天", temperature: "28°C ~ 32°C", condition: "中雨" },
      { date: "大后天", temperature: "29°C ~ 33°C", condition: "多云" },
    ],
  },
  深圳: {
    temperature: 31,
    condition: "阴天",
    humidity: 75,
    windSpeed: 10,
    aqi: 48,
    forecast: [
      { date: "明天", temperature: "28°C ~ 32°C", condition: "小雨" },
      { date: "后天", temperature: "27°C ~ 31°C", condition: "中雨" },
      { date: "大后天", temperature: "28°C ~ 32°C", condition: "多云" },
    ],
  },
  杭州: {
    temperature: 26,
    condition: "小雨",
    humidity: 70,
    windSpeed: 6,
    aqi: 60,
    forecast: [
      { date: "明天", temperature: "24°C ~ 28°C", condition: "中雨" },
      { date: "后天", temperature: "23°C ~ 27°C", condition: "小雨" },
      { date: "大后天", temperature: "24°C ~ 29°C", condition: "多云" },
    ],
  },
  成都: {
    temperature: 24,
    condition: "阴天",
    humidity: 78,
    windSpeed: 5,
    aqi: 95,
    forecast: [
      { date: "明天", temperature: "22°C ~ 26°C", condition: "小雨" },
      { date: "后天", temperature: "21°C ~ 25°C", condition: "阴天" },
      { date: "大后天", temperature: "22°C ~ 27°C", condition: "多云" },
    ],
  },
  西安: {
    temperature: 27,
    condition: "晴朗",
    humidity: 50,
    windSpeed: 9,
    aqi: 110,
    forecast: [
      { date: "明天", temperature: "25°C ~ 30°C", condition: "晴朗" },
      { date: "后天", temperature: "24°C ~ 29°C", condition: "多云" },
      { date: "大后天", temperature: "23°C ~ 28°C", condition: "小雨" },
    ],
  },
};

/**
 * 获取空气质量等级描述
 */
function getAQIDescription(aqi: number): string {
  if (aqi <= 50) return "优";
  if (aqi <= 100) return "良";
  if (aqi <= 150) return "轻度污染";
  if (aqi <= 200) return "中度污染";
  if (aqi <= 300) return "重度污染";
  return "严重污染";
}

/**
 * 创建 MCP 服务器实例
 */
const server = new Server(
  {
    name: "weather-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {},
    },
  }
);

/**
 * 处理 ListTools 请求
 * 返回天气服务器支持的所有工具
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "get_current_weather",
        description: "获取指定城市的当前天气信息，包括温度、天气状况、湿度和风速",
        inputSchema: {
          type: "object",
          properties: {
            city: {
              type: "string",
              description: "城市名称，如：北京、上海、广州、深圳、杭州、成都、西安",
            },
          },
          required: ["city"],
        },
      },
      {
        name: "get_weather_forecast",
        description: "获取指定城市未来几天的天气预报",
        inputSchema: {
          type: "object",
          properties: {
            city: {
              type: "string",
              description: "城市名称，如：北京、上海、广州、深圳、杭州、成都、西安",
            },
            days: {
              type: "number",
              description: "预报天数（1-3天），默认为3天",
            },
          },
          required: ["city"],
        },
      },
      {
        name: "get_air_quality",
        description: "获取指定城市的空气质量指数（AQI）和等级",
        inputSchema: {
          type: "object",
          properties: {
            city: {
              type: "string",
              description: "城市名称，如：北京、上海、广州、深圳、杭州、成都、西安",
            },
          },
          required: ["city"],
        },
      },
      {
        name: "get_all_cities",
        description: "获取支持查询的所有城市列表",
        inputSchema: {
          type: "object",
          properties: {},
        },
      },
    ],
  };
});

/**
 * 处理 CallTool 请求
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "get_current_weather": {
      const { city } = args as { city: string };
      const data = weatherDatabase[city];

      if (!data) {
        return {
          content: [
            {
              type: "text",
              text: `抱歉，暂不支持查询 ${city} 的天气信息。\n支持的城市：${Object.keys(weatherDatabase).join("、")}`,
            },
          ],
        };
      }

      return {
        content: [
          {
            type: "text",
            text: `${city}当前天气：\n` +
                  `• 天气状况：${data.condition}\n` +
                  `• 温度：${data.temperature}°C\n` +
                  `• 湿度：${data.humidity}%\n` +
                  `• 风速：${data.windSpeed} km/h`,
          },
        ],
      };
    }

    case "get_weather_forecast": {
      const { city, days = 3 } = args as { city: string; days?: number };
      const data = weatherDatabase[city];

      if (!data) {
        return {
          content: [
            {
              type: "text",
              text: `抱歉，暂不支持查询 ${city} 的天气预报。\n支持的城市：${Object.keys(weatherDatabase).join("、")}`,
            },
          ],
        };
      }

      const forecastDays = Math.min(Math.max(days, 1), 3);
      const forecast = data.forecast.slice(0, forecastDays);

      let forecastText = `${city}未来${forecastDays}天天气预报：\n`;
      forecast.forEach((day, index) => {
        forecastText += `${index + 1}. ${day.date}：${day.condition}，${day.temperature}\n`;
      });

      return {
        content: [
          {
            type: "text",
            text: forecastText.trim(),
          },
        ],
      };
    }

    case "get_air_quality": {
      const { city } = args as { city: string };
      const data = weatherDatabase[city];

      if (!data) {
        return {
          content: [
            {
              type: "text",
              text: `抱歉，暂不支持查询 ${city} 的空气质量。\n支持的城市：${Object.keys(weatherDatabase).join("、")}`,
            },
          ],
        };
      }

      const aqiLevel = getAQIDescription(data.aqi);

      return {
        content: [
          {
            type: "text",
            text: `${city}空气质量：\n` +
                  `• AQI指数：${data.aqi}\n` +
                  `• 空气质量等级：${aqiLevel}`,
          },
        ],
      };
    }

    case "get_all_cities": {
      return {
        content: [
          {
            type: "text",
            text: `支持查询的城市列表：\n${Object.keys(weatherDatabase).join("、")}`,
          },
        ],
      };
    }

    default:
      throw new Error(`未知工具: ${name}`);
  }
});

/**
 * 主函数：启动 MCP 服务器
 */
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("天气 MCP 服务器已启动，通过 stdio 传输方式运行");
}

main().catch((error) => {
  console.error("服务器启动失败:", error);
  process.exit(1);
});
