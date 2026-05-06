/**
 * ============================================
 * LangChain MCP (Model Context Protocol) 功能演示
 * ============================================
 *
 * 本演示展示了如何使用 LangChain 与 MCP (Model Context Protocol) 集成
 * MCP 是 Anthropic 推出的开放协议，用于标准化应用程序向 LLM 提供工具和上下文的方式
 *
 * 本 demo 包含以下内容：
 *   1. MCP 基础概念介绍
 *   2. 使用 @langchain/mcp-adapters 连接 MCP 服务器
 *   3. 通过 stdio 传输方式连接本地 MCP 服务器
 *   4. 使用 MCP 工具进行数学计算和天气查询
 *   5. 多服务器 MCP 客户端配置
 *   6. 创建智能体并调用 MCP 工具
 *
 * 前置要求：
 *   - 安装依赖：pnpm add @langchain/mcp-adapters @modelcontextprotocol/sdk
 *   - 配置环境变量：DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
 */

import { ChatOpenAI } from "@langchain/openai";
import { MultiServerMCPClient } from "@langchain/mcp-adapters";
import { HumanMessage, AIMessage, ToolMessage, BaseMessage } from "@langchain/core/messages";
import dotenv from "dotenv";
import { fileURLToPath } from "url";
import { dirname, join } from "path";

// 获取当前文件的目录路径（ESM 环境下替代 __dirname）
const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// 加载环境变量
// .env 文件应包含：
//   DEEPSEEK_API_KEY=your_api_key
//   DEEPSEEK_BASE_URL=https://api.deepseek.com
//   DEEPSEEK_MODEL=deepseek-chat
dotenv.config();

/**
 * ============================================
 * 第一部分：MCP 基础概念
 * ============================================
 *
 * MCP (Model Context Protocol) 是什么？
 * ----------------------------------------
 * MCP 是一种开放协议，标准化了应用程序如何向大语言模型（LLM）提供：
 *   - Tools（工具）：AI 可调用的功能
 *   - Resources（资源）：AI 可读取的数据
 *   - Prompts（提示词）：预定义的提示模板
 *
 * MCP 的核心优势：
 *   1. 标准化：统一的协议，无需为每个工具写适配代码
 *   2. 可组合：可以轻松组合多个 MCP 服务器的工具
 *   3. 安全：工具在独立进程中运行，与主应用隔离
 *   4. 灵活：支持多种传输方式（stdio、HTTP、SSE）
 *
 * MCP 传输方式：
 *   - stdio：标准输入输出，适合本地工具，服务器作为子进程运行
 *   - Streamable HTTP：HTTP 流式传输，支持远程连接
 *   - SSE：Server-Sent Events，适合实时推送场景
 */

/**
 * ============================================
 * 第二部分：初始化 LangChain 模型
 * ============================================
 *
 * 这里使用 DeepSeek API 作为 LLM 后端
 * 模型需要支持工具调用（function calling）才能使用 MCP 工具
 */
const model = new ChatOpenAI({
  modelName: process.env.DEEPSEEK_MODEL,
  configuration: {
    baseURL: process.env.DEEPSEEK_BASE_URL,
    apiKey: process.env.DEEPSEEK_API_KEY,
    timeout: 60000,
  },
  temperature: 0.7,
  maxRetries: 2,
});

/**
 * ============================================
 * 第三部分：创建多服务器 MCP 客户端
 * ============================================
 *
 * MultiServerMCPClient 允许同时连接多个 MCP 服务器
 * 每个服务器可以暴露不同的工具集
 *
 * 配置说明：
 *   - math: 数学计算服务器（本地 stdio 传输）
 *   - weather: 天气查询服务器（本地 stdio 传输）
 */
async function createMCPClient(): Promise<MultiServerMCPClient> {
  // 构建 MCP 服务器文件的绝对路径
  const mathServerPath = join(__dirname, "mathServer.ts");
  const weatherServerPath = join(__dirname, "weatherServer.ts");

  const client = new MultiServerMCPClient({
    // 数学服务器配置
    math: {
      transport: "stdio", // 使用标准输入输出传输
      command: "npx",     // 使用 npx 运行 TypeScript 文件
      args: [
        "tsx",            // 使用 tsx 直接执行 TypeScript
        mathServerPath,   // 服务器文件路径
      ],
    },
    // 天气服务器配置
    weather: {
      transport: "stdio",
      command: "npx",
      args: [
        "tsx",
        weatherServerPath,
      ],
    },
  });

  return client;
}

/**
 * ============================================
 * 第四部分：工具调用核心逻辑
 * ============================================
 *
 * 处理流程：
 *   1. 用户输入 -> 模型判断是否需要调用工具
 *   2. 如果需要，模型输出 tool_calls（工具名和参数）
 *   3. 通过 MCP 客户端调用对应的 MCP 服务器工具
 *   4. 将工具执行结果返回给模型
 *   5. 模型基于工具结果生成最终回答
 *
 * @param client MCP 客户端实例
 * @param userInput 用户输入的文本
 * @returns AI 的最终回答
 */
async function processWithMCPTools(
  client: MultiServerMCPClient,
  userInput: string
): Promise<string> {
  // 第一步：从所有 MCP 服务器获取工具
  // getTools() 会自动收集所有已连接服务器的工具
  const tools = await client.getTools();
  console.log(`[系统] 已加载 ${tools.length} 个 MCP 工具`);

  // 显示可用工具列表（调试用）
  tools.forEach((tool, index) => {
    console.log(`  ${index + 1}. ${tool.name}: ${tool.description}`);
  });

  // 第二步：将工具绑定到模型
  // 这样模型就知道可以使用哪些工具
  const modelWithTools = model.bindTools(tools);

  // 第三步：创建消息列表，开始对话
  const messages: BaseMessage[] = [new HumanMessage(userInput)];

  // 第四步：调用模型，让它决定是否使用工具
  const aiMessage = await modelWithTools.invoke(messages);

  // 第五步：检查模型是否请求了工具调用
  if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
    console.log(`[工具调用] 模型决定调用 ${aiMessage.tool_calls.length} 个工具`);

    // 将 AI 的消息（包含 tool_calls）添加到消息列表
    messages.push(aiMessage);

    // 第六步：依次执行每个工具调用
    for (const toolCall of aiMessage.tool_calls) {
      console.log(`  → 调用工具：${toolCall.name}`);
      console.log(`    参数：${JSON.stringify(toolCall.args)}`);

      // 在工具列表中查找对应的工具
      const tool = tools.find((t) => t.name === toolCall.name);

      if (!tool) {
        console.error(`    错误：找不到工具 ${toolCall.name}`);
        continue;
      }

      try {
        // 执行工具调用
        const toolResult = await tool.invoke(toolCall.args);
        console.log(`    结果：${toolResult}`);

        // 将工具执行结果封装为 ToolMessage 添加到消息列表
        messages.push(
          new ToolMessage({
            content: String(toolResult),
            tool_call_id: toolCall.id || "unknown",
          })
        );
      } catch (error) {
        console.error(`    工具执行出错：${error}`);
        messages.push(
          new ToolMessage({
            content: `工具执行出错：${error}`,
            tool_call_id: toolCall.id || "unknown",
          })
        );
      }
    }

    // 第七步：再次调用模型，传入工具执行结果
    // 模型会基于工具结果生成最终的自然语言回答
    const finalResponse = await modelWithTools.invoke(messages);
    return String(finalResponse.content);
  }

  // 如果没有工具调用，直接返回模型的回答
  return String(aiMessage.content);
}

/**
 * ============================================
 * 第五部分：各种演示场景
 * ============================================
 */

/**
 * 演示 1：基础 MCP 工具调用
 * 展示单个 MCP 工具的调用流程
 */
async function runBasicMCPDemo(client: MultiServerMCPClient) {
  console.log("\n========================================");
  console.log("演示 1：基础 MCP 工具调用");
  console.log("========================================\n");

  const testCases = [
    { input: "计算 123 + 456 等于多少？", expectedTool: "数学计算" },
    { input: "北京今天天气怎么样？", expectedTool: "天气查询" },
    { input: "25 的平方根是多少？", expectedTool: "平方根计算" },
    { input: "上海未来3天的天气预报", expectedTool: "天气预报" },
  ];

  for (const testCase of testCases) {
    console.log(`\n用户：${testCase.input}`);
    console.log(`[预期调用：${testCase.expectedTool}工具]`);

    try {
      const response = await processWithMCPTools(client, testCase.input);
      console.log(`AI：${response}`);
    } catch (error) {
      console.error(`错误：${error}`);
    }

    // 添加延迟，避免请求过快
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

/**
 * 演示 2：多步骤计算
 * 展示 AI 如何组合多个 MCP 工具完成复杂任务
 */
async function runMultiStepDemo(client: MultiServerMCPClient) {
  console.log("\n========================================");
  console.log("演示 2：多步骤计算");
  console.log("========================================\n");

  const complexQueries = [
    "计算 (100 + 200) * 3，然后对结果开平方根",
    "北京和上海哪个城市温度更高？相差多少度？",
    "如果我有 1000 元，买 3 件 256 元的商品，还剩多少钱？",
  ];

  for (const query of complexQueries) {
    console.log(`\n用户：${query}`);

    try {
      const response = await processWithMCPTools(client, query);
      console.log(`AI：${response}`);
    } catch (error) {
      console.error(`错误：${error}`);
    }

    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}

/**
 * 演示 3：带对话历史的 MCP 工具调用
 * 展示如何在多轮对话中使用 MCP 工具
 */
async function runConversationalDemo(client: MultiServerMCPClient) {
  console.log("\n========================================");
  console.log("演示 3：对话式 MCP 工具调用");
  console.log("========================================\n");

  // 获取工具并绑定到模型
  const tools = await client.getTools();
  const modelWithTools = model.bindTools(tools);

  // 存储对话历史
  const messages: BaseMessage[] = [];

  // 定义对话流程
  const conversation = [
    "你好，我想知道北京的天气。",
    "那儿的空气质量怎么样？",
    "帮我计算一下，如果北京今天温度是25度，明天气温下降3度，后天又上升5度，后天温度是多少？",
    "25度的平方根约等于多少？",
  ];

  for (const userInput of conversation) {
    console.log(`\n用户：${userInput}`);

    // 添加用户消息到历史
    messages.push(new HumanMessage(userInput));

    try {
      // 调用模型（带工具）
      const aiMessage = await modelWithTools.invoke(messages);

      // 检查是否需要工具调用
      if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
        messages.push(aiMessage);

        // 执行工具调用
        for (const toolCall of aiMessage.tool_calls) {
          const tool = tools.find((t) => t.name === toolCall.name);
          if (tool) {
            const toolResult = await tool.invoke(toolCall.args);
            messages.push(
              new ToolMessage({
                content: String(toolResult),
                tool_call_id: toolCall.id || "unknown",
              })
            );
          }
        }

        // 获取最终回答
        const finalResponse = await modelWithTools.invoke(messages);
        console.log(`AI：${finalResponse.content}`);
        messages.push(new AIMessage(String(finalResponse.content)));
      } else {
        console.log(`AI：${aiMessage.content}`);
        messages.push(new AIMessage(String(aiMessage.content)));
      }
    } catch (error) {
      console.error(`错误：${error}`);
    }

    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

/**
 * 演示 4：工具组合使用
 * 展示如何在一个查询中使用多个 MCP 服务器的工具
 */
async function runToolCombinationDemo(client: MultiServerMCPClient) {
  console.log("\n========================================");
  console.log("演示 4：工具组合使用");
  console.log("========================================\n");

  const combinationQueries = [
    "北京和广州的温度相差多少度？帮我计算一下差值。",
    "上海现在的温度是28度，如果温度上升 2 的 3 次方度，最终温度是多少？",
    "查询深圳和杭州的天气，然后告诉我哪个城市更适合户外活动（温度适中、空气质量好）？",
  ];

  for (const query of combinationQueries) {
    console.log(`\n用户：${query}`);

    try {
      const response = await processWithMCPTools(client, query);
      console.log(`AI：${response}`);
    } catch (error) {
      console.error(`错误：${error}`);
    }

    await new Promise((resolve) => setTimeout(resolve, 1500));
  }
}

/**
 * 演示 5：错误处理
 * 展示如何处理 MCP 工具调用中的错误情况
 */
async function runErrorHandlingDemo(client: MultiServerMCPClient) {
  console.log("\n========================================");
  console.log("演示 5：错误处理");
  console.log("========================================\n");

  const errorCases = [
    "查询火星的天气", // 不存在的城市
    "计算 100 除以 0", // 除零错误
    "查询纽约的温度", // 不支持的城市
  ];

  for (const input of errorCases) {
    console.log(`\n用户：${input}`);

    try {
      const response = await processWithMCPTools(client, input);
      console.log(`AI：${response}`);
    } catch (error) {
      console.error(`捕获到错误：${error}`);
    }

    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

/**
 * ============================================
 * 第六部分：主函数
 * ============================================
 */
async function main() {
  console.log("LangChain MCP (Model Context Protocol) 功能演示");
  console.log("================================================\n");

  let client: MultiServerMCPClient | null = null;

  try {
    // 创建 MCP 客户端
    console.log("[系统] 正在初始化 MCP 客户端...");
    client = await createMCPClient();
    console.log("[系统] MCP 客户端初始化完成\n");

    // 运行基础 MCP 工具调用演示
    await runBasicMCPDemo(client);

    // 运行多步骤计算演示
    await runMultiStepDemo(client);

    // 运行对话式 MCP 工具调用演示
    await runConversationalDemo(client);

    // 运行工具组合使用演示
    await runToolCombinationDemo(client);

    // 运行错误处理演示
    await runErrorHandlingDemo(client);

    console.log("\n========================================");
    console.log("所有演示运行完成！");
    console.log("========================================");
  } catch (error) {
    console.error("运行出错：", error);
  } finally {
    // 关闭 MCP 客户端，清理资源
    if (client) {
      console.log("\n[系统] 正在关闭 MCP 客户端...");
      await client.close();
      console.log("[系统] MCP 客户端已关闭");
    }
  }
}

// 启动程序
main();
