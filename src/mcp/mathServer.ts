/**
 * ============================================
 * MCP 数学服务器 (Math MCP Server)
 * ============================================
 *
 * 这是一个基于 Model Context Protocol (MCP) 的数学计算服务器
 * 它通过 stdio 传输方式暴露数学工具，供 LangChain 智能体调用
 *
 * MCP (Model Context Protocol) 是 Anthropic 推出的开放协议，
 * 用于标准化应用程序向大语言模型提供工具和上下文的方式。
 *
 * 运行方式：
 *   node mathServer.ts
 *
 * 或者通过 MCP 客户端启动作为子进程
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import {
  CallToolRequestSchema,
  ListToolsRequestSchema,
} from "@modelcontextprotocol/sdk/types.js";

/**
 * 创建 MCP 服务器实例
 * name: 服务器名称
 * version: 服务器版本
 * capabilities: 声明服务器支持的能力（这里是 tools）
 */
const server = new Server(
  {
    name: "math-server",
    version: "1.0.0",
  },
  {
    capabilities: {
      tools: {}, // 声明支持工具调用能力
    },
  }
);

/**
 * 处理 ListTools 请求
 * 当客户端请求可用工具列表时，返回所有可用的工具定义
 * 每个工具包含：
 *   - name: 工具名称（唯一标识）
 *   - description: 工具描述（帮助 AI 理解何时使用该工具）
 *   - inputSchema: 输入参数的 JSON Schema 定义
 */
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return {
    tools: [
      {
        name: "add",
        description: "将两个数字相加，返回它们的和",
        inputSchema: {
          type: "object",
          properties: {
            a: {
              type: "number",
              description: "第一个加数",
            },
            b: {
              type: "number",
              description: "第二个加数",
            },
          },
          required: ["a", "b"],
        },
      },
      {
        name: "subtract",
        description: "计算两个数字的差（第一个数减去第二个数）",
        inputSchema: {
          type: "object",
          properties: {
            a: {
              type: "number",
              description: "被减数",
            },
            b: {
              type: "number",
              description: "减数",
            },
          },
          required: ["a", "b"],
        },
      },
      {
        name: "multiply",
        description: "将两个数字相乘，返回它们的积",
        inputSchema: {
          type: "object",
          properties: {
            a: {
              type: "number",
              description: "第一个乘数",
            },
            b: {
              type: "number",
              description: "第二个乘数",
            },
          },
          required: ["a", "b"],
        },
      },
      {
        name: "divide",
        description: "将第一个数字除以第二个数字，返回商",
        inputSchema: {
          type: "object",
          properties: {
            a: {
              type: "number",
              description: "被除数",
            },
            b: {
              type: "number",
              description: "除数（不能为0）",
            },
          },
          required: ["a", "b"],
        },
      },
      {
        name: "power",
        description: "计算一个数的幂次方",
        inputSchema: {
          type: "object",
          properties: {
            base: {
              type: "number",
              description: "底数",
            },
            exponent: {
              type: "number",
              description: "指数",
            },
          },
          required: ["base", "exponent"],
        },
      },
      {
        name: "sqrt",
        description: "计算一个数的平方根",
        inputSchema: {
          type: "object",
          properties: {
            number: {
              type: "number",
              description: "要计算平方根的数字（必须大于等于0）",
            },
          },
          required: ["number"],
        },
      },
    ],
  };
});

/**
 * 处理 CallTool 请求
 * 当客户端调用某个工具时，执行对应的逻辑并返回结果
 * request.params.name: 被调用的工具名称
 * request.params.arguments: 工具调用参数
 */
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;

  switch (name) {
    case "add": {
      const { a, b } = args as { a: number; b: number };
      const result = a + b;
      return {
        content: [
          {
            type: "text",
            text: String(result),
          },
        ],
      };
    }

    case "subtract": {
      const { a, b } = args as { a: number; b: number };
      const result = a - b;
      return {
        content: [
          {
            type: "text",
            text: String(result),
          },
        ],
      };
    }

    case "multiply": {
      const { a, b } = args as { a: number; b: number };
      const result = a * b;
      return {
        content: [
          {
            type: "text",
            text: String(result),
          },
        ],
      };
    }

    case "divide": {
      const { a, b } = args as { a: number; b: number };
      if (b === 0) {
        return {
          content: [
            {
              type: "text",
              text: "错误：除数不能为0",
            },
          ],
          isError: true,
        };
      }
      const result = a / b;
      return {
        content: [
          {
            type: "text",
            text: String(result),
          },
        ],
      };
    }

    case "power": {
      const { base, exponent } = args as { base: number; exponent: number };
      const result = Math.pow(base, exponent);
      return {
        content: [
          {
            type: "text",
            text: String(result),
          },
        ],
      };
    }

    case "sqrt": {
      const { number } = args as { number: number };
      if (number < 0) {
        return {
          content: [
            {
              type: "text",
              text: "错误：不能计算负数的平方根",
            },
          ],
          isError: true,
        };
      }
      const result = Math.sqrt(number);
      return {
        content: [
          {
            type: "text",
            text: String(result),
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
 * 使用 StdioServerTransport 通过标准输入输出进行通信
 */
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  // 输出到 stderr，避免干扰 stdio 通信
  console.error("数学 MCP 服务器已启动，通过 stdio 传输方式运行");
}

main().catch((error) => {
  console.error("服务器启动失败:", error);
  process.exit(1);
});
