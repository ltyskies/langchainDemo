/**
 * ============================================
 * LangChain Agent Loop (智能体循环) 功能演示
 * ============================================
 *
 * 本演示展示了如何使用 LangChain 构建一个完整的 Agent Loop（智能体循环）
 * Agent Loop 是 AI 智能体的核心机制，它允许 AI 自主地思考、决策、执行工具，
 * 并根据结果继续思考，形成一个循环，直到完成任务。
 *
 * 本 demo 包含以下内容：
 *   1. Agent Loop 核心概念介绍
 *   2. 手动构建 Agent Loop（不使用 LangGraph）
 *   3. ReAct (Reasoning + Acting) 模式实现
 *   4. 多轮工具调用与推理循环
 *   5. 记忆管理与对话历史
 *   6. 循环终止条件与最大迭代控制
 *   7. 流式输出与实时反馈
 *
 * Agent Loop 的核心流程：
 *   1. 用户输入 -> Agent 开始思考
 *   2. Agent 决定是否需要调用工具
 *   3. 如果需要，执行工具并获取观察结果
 *   4. Agent 基于观察结果继续思考
 *   5. 重复步骤 2-4，直到 Agent 决定给出最终答案
 *   6. 返回最终答案给用户
 *
 * 前置要求：
 *   - 安装依赖：pnpm add @langchain/core
 *   - 配置环境变量：DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
 */

import { ChatOpenAI } from "@langchain/openai";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import { HumanMessage, AIMessage, ToolMessage, SystemMessage, type BaseMessage, isAIMessage } from "@langchain/core/messages";
import dotenv from "dotenv";

// 加载环境变量
dotenv.config();

/**
 * ============================================
 * 第一部分：Agent Loop 核心概念
 * ============================================
 *
 * 什么是 Agent Loop？
 * ------------------
 * Agent Loop 是一种让 AI 智能体能够自主决策和行动的循环机制。
 * 与传统的单次调用不同，Agent Loop 允许 AI 进行多轮思考和工具调用，
 * 直到完成复杂的任务。
 *
 * Agent Loop 的关键组件：
 *   1. 思考 (Thought)：AI 分析当前情况，决定下一步行动
 *   2. 行动 (Action)：AI 调用工具获取信息或执行操作
 *   3. 观察 (Observation)：AI 接收工具执行的结果
 *   4. 循环 (Loop)：重复思考-行动-观察，直到任务完成
 *
 * ReAct 模式：
 *   ReAct = Reasoning (推理) + Acting (行动)
 *   这是目前最流行的 Agent 架构，AI 在每一步都会明确输出思考过程，
 *   然后基于思考结果采取行动。
 */

/**
 * ============================================
 * 第二部分：工具定义
 * ============================================
 *
 * 定义 Agent 可以使用的工具集合
 * 这些工具是 Agent 与外部世界交互的接口
 */

/**
 * 工具 1：搜索引擎工具
 * 模拟搜索互联网获取信息
 */
const searchTool = tool(
  async ({ query }: { query: string }) => {
    // 模拟搜索结果（实际应用中应调用真实搜索 API）
    const searchResults: Record<string, string> = {
      "LangChain": "LangChain 是一个用于构建 LLM 应用程序的框架，提供了链、代理、记忆等组件。",
      "OpenAI": "OpenAI 是一家人工智能研究公司，开发了 GPT 系列大语言模型。",
      "DeepSeek": "DeepSeek 是一家中国 AI 公司，开发了 DeepSeek-V3 等大语言模型。",
      "React": "React 是 Facebook 开发的用于构建用户界面的 JavaScript 库。",
      "TypeScript": "TypeScript 是 JavaScript 的超集，添加了静态类型系统。",
    };

    // 查找匹配的结果
    for (const [key, value] of Object.entries(searchResults)) {
      if (query.toLowerCase().includes(key.toLowerCase())) {
        return `搜索结果：${value}`;
      }
    }

    return `搜索结果：找到关于 "${query}" 的相关信息。这是一个模拟搜索结果。`;
  },
  {
    name: "search",
    description: "搜索互联网获取信息，用于查找最新数据、事实、定义等",
    schema: z.object({
      query: z.string().describe("搜索关键词或问题"),
    }),
  }
);

/**
 * 工具 2：计算器工具
 * 执行数学计算
 */
const calculatorTool = tool(
  async ({ expression }: { expression: string }) => {
    try {
      // 安全地计算数学表达式
      const sanitizedExpression = expression.replace(/[^0-9+\-*/.()\s]/g, "");
      const result = new Function(`return ${sanitizedExpression}`)();
      return `计算结果：${result}`;
    } catch (error) {
      return `计算错误：无法计算 "${expression}"`;
    }
  },
  {
    name: "calculator",
    description: "执行数学计算，支持加减乘除和括号",
    schema: z.object({
      expression: z.string().describe("数学表达式，如：2 + 3 * 4"),
    }),
  }
);

/**
 * 工具 3：天气查询工具
 * 查询指定城市的天气
 */
const weatherTool = tool(
  async ({ city }: { city: string }) => {
    const weatherData: Record<string, { temp: number; condition: string }> = {
      "北京": { temp: 25, condition: "晴朗" },
      "上海": { temp: 28, condition: "多云" },
      "广州": { temp: 32, condition: "雷阵雨" },
      "深圳": { temp: 31, condition: "阴天" },
    };

    const weather = weatherData[city];
    if (weather) {
      return `${city}天气：${weather.condition}，温度 ${weather.temp}°C`;
    }
    return `暂无 ${city} 的天气信息`;
  },
  {
    name: "get_weather",
    description: "查询指定城市的当前天气",
    schema: z.object({
      city: z.string().describe("城市名称，如：北京、上海"),
    }),
  }
);

/**
 * 工具 4：代码执行工具
 * 模拟执行代码并返回结果
 */
const codeExecutionTool = tool(
  async ({ code, language }: { code: string; language: string }) => {
    // 模拟代码执行结果
    if (language === "python") {
      if (code.includes("print")) {
        const match = code.match(/print\(["'](.+)["']\)/);
        return `代码执行结果：输出 "${match?.[1] || "Hello World"}"`;
      }
      return "代码执行结果：Python 代码执行成功";
    }
    return `代码执行结果：${language} 代码已执行`;
  },
  {
    name: "execute_code",
    description: "执行代码并返回结果，支持 Python、JavaScript 等语言",
    schema: z.object({
      code: z.string().describe("要执行的代码"),
      language: z.string().describe("编程语言，如：python、javascript"),
    }),
  }
);

// 工具集合
const tools = [searchTool, calculatorTool, weatherTool, codeExecutionTool];

// 创建工具名称到工具执行函数的映射
// 注意：这里直接调用工具函数，而不是使用 tool.invoke()
const toolsByName: Record<string, (args: Record<string, any>) => Promise<string>> = {
  search: async (args) => {
    const searchResults: Record<string, string> = {
      "LangChain": "LangChain 是一个用于构建 LLM 应用程序的框架，提供了链、代理、记忆等组件。",
      "OpenAI": "OpenAI 是一家人工智能研究公司，开发了 GPT 系列大语言模型。",
      "DeepSeek": "DeepSeek 是一家中国 AI 公司，开发了 DeepSeek-V3 等大语言模型。",
      "React": "React 是 Facebook 开发的用于构建用户界面的 JavaScript 库。",
      "TypeScript": "TypeScript 是 JavaScript 的超集，添加了静态类型系统。",
    };
    const query = args.query as string;
    for (const [key, value] of Object.entries(searchResults)) {
      if (query.toLowerCase().includes(key.toLowerCase())) {
        return `搜索结果：${value}`;
      }
    }
    return `搜索结果：找到关于 "${query}" 的相关信息。这是一个模拟搜索结果。`;
  },
  calculator: async (args) => {
    try {
      const expression = (args.expression as string).replace(/[^0-9+\-*/.()\s]/g, "");
      const result = new Function(`return ${expression}`)();
      return `计算结果：${result}`;
    } catch (error) {
      return `计算错误：无法计算 "${args.expression}"`;
    }
  },
  get_weather: async (args) => {
    const weatherData: Record<string, { temp: number; condition: string }> = {
      "北京": { temp: 25, condition: "晴朗" },
      "上海": { temp: 28, condition: "多云" },
      "广州": { temp: 32, condition: "雷阵雨" },
      "深圳": { temp: 31, condition: "阴天" },
    };
    const city = args.city as string;
    const weather = weatherData[city];
    if (weather) {
      return `${city}天气：${weather.condition}，温度 ${weather.temp}°C`;
    }
    return `暂无 ${city} 的天气信息`;
  },
  execute_code: async (args) => {
    const code = args.code as string;
    const language = args.language as string;
    if (language === "python") {
      if (code.includes("print")) {
        const match = code.match(/print\(["'](.+)["']\)/);
        return `代码执行结果：输出 "${match?.[1] || "Hello World"}"`;
      }
      return "代码执行结果：Python 代码执行成功";
    }
    return `代码执行结果：${language} 代码已执行`;
  },
};

/**
 * ============================================
 * 第三部分：模型初始化
 * ============================================
 */

/**
 * 初始化 ChatOpenAI 模型
 * 这里使用 DeepSeek API
 */
const model = new ChatOpenAI({
  model: process.env.DEEPSEEK_MODEL || "deepseek-chat",
  apiKey: process.env.DEEPSEEK_API_KEY || "",
  configuration: {
    baseURL: process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com",
  },
  temperature: 0.7,
  maxRetries: 2,
});

/**
 * 将工具绑定到模型
 * 这样模型就知道可以使用哪些工具
 */
const modelWithTools = model.bindTools(tools);

/**
 * ============================================
 * 第四部分：系统提示词
 * ============================================
 */

/**
 * 系统提示词
 * 定义 Agent 的行为准则和思考方式
 */
const SYSTEM_PROMPT = `你是一个智能助手，可以帮助用户完成各种任务。

你可以使用以下工具：
1. search - 搜索互联网获取信息
2. calculator - 执行数学计算
3. get_weather - 查询天气
4. execute_code - 执行代码

工作流程：
1. 分析用户的问题，确定是否需要使用工具
2. 如果需要，选择合适的工具并调用
3. 根据工具返回的结果，决定下一步行动
4. 重复上述过程，直到获得足够信息回答问题
5. 给出最终答案

注意事项：
- 如果需要多个工具，可以多次调用
- 仔细分析工具返回的结果
- 如果一次搜索没有找到答案，可以尝试不同的关键词`;

/**
 * ============================================
 * 第五部分：Agent Loop 核心实现
 * ============================================
 *
 * 这是 Agent Loop 的核心逻辑，包含：
 *   - 思考节点：调用 LLM 进行思考
 *   - 工具执行：执行工具调用
 *   - 循环控制：决定是否继续循环
 */

/**
 * Agent Loop 配置选项
 */
interface AgentLoopOptions {
  maxIterations?: number;  // 最大迭代次数，防止无限循环
  verbose?: boolean;       // 是否输出详细日志
}

/**
 * Agent Loop 结果
 */
interface AgentLoopResult {
  finalAnswer: string;     // 最终答案
  messages: BaseMessage[]; // 完整的对话历史
  iterationCount: number;  // 迭代次数
}

/**
 * 执行 Agent Loop
 * 这是 Agent 的核心循环逻辑
 *
 * @param userInput 用户输入
 * @param options 配置选项
 * @returns Agent 执行结果
 */
async function runAgentLoop(
  userInput: string,
  options: AgentLoopOptions = {}
): Promise<AgentLoopResult> {
  const { maxIterations = 10, verbose = true } = options;

  // 初始化消息历史
  const messages: BaseMessage[] = [
    new SystemMessage(SYSTEM_PROMPT),
    new HumanMessage(userInput),
  ];

  let iterationCount = 0;

  if (verbose) {
    console.log("\n" + "=".repeat(50));
    console.log("开始 Agent Loop");
    console.log("用户输入：", userInput);
    console.log("=".repeat(50));
  }

  // Agent Loop 主循环
  while (iterationCount < maxIterations) {
    iterationCount++;

    if (verbose) {
      console.log(`\n--- 迭代 ${iterationCount} ---`);
    }

    // 步骤 1：Agent 思考
    if (verbose) {
      console.log("[1] Agent 思考中...");
    }

    const aiMessage = await modelWithTools.invoke(messages);
    messages.push(aiMessage);

    // 步骤 2：检查是否需要调用工具
    if (isAIMessage(aiMessage) && aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
      if (verbose) {
        console.log(`[2] Agent 决定调用 ${aiMessage.tool_calls.length} 个工具：`);
      }

      // 步骤 3：执行工具
      for (const toolCall of aiMessage.tool_calls) {
        if (verbose) {
          console.log(`    - ${toolCall.name}: ${JSON.stringify(toolCall.args)}`);
        }

        const toolFunc = toolsByName[toolCall.name];
        if (toolFunc) {
          const result = await toolFunc(toolCall.args);

          if (verbose) {
            console.log(`    结果: ${result}`);
          }

          // 步骤 4：添加观察结果（ToolMessage）
          messages.push(
            new ToolMessage({
              content: result,
              tool_call_id: toolCall.id ?? "unknown",
            })
          );
        } else {
          // 工具不存在
          const errorMsg = `错误：未知工具 "${toolCall.name}"`;
          if (verbose) {
            console.log(`    ${errorMsg}`);
          }
          messages.push(
            new ToolMessage({
              content: errorMsg,
              tool_call_id: toolCall.id ?? "unknown",
            })
          );
        }
      }

      // 继续下一轮循环
      if (verbose) {
        console.log("[3] 继续下一轮思考...");
      }
    } else {
      // Agent 给出了最终答案
      const finalAnswer = String(aiMessage.content);

      if (verbose) {
        console.log("[2] Agent 给出最终答案：");
        console.log(`    ${finalAnswer}`);
        console.log(`\nAgent Loop 结束，共 ${iterationCount} 轮迭代`);
      }

      return {
        finalAnswer,
        messages,
        iterationCount,
      };
    }
  }

  // 达到最大迭代次数
  if (verbose) {
    console.log(`\n[警告] 达到最大迭代次数 (${maxIterations})，强制结束`);
  }

  const lastMessage = messages[messages.length - 1];
  return {
    finalAnswer: String(lastMessage?.content ?? "达到最大迭代次数，无法给出答案"),
    messages,
    iterationCount,
  };
}

/**
 * 带记忆的 Agent Loop
 * 保存对话历史，支持多轮对话
 */
class ConversationalAgent {
  private messages: BaseMessage[] = [];
  private maxIterations: number;
  private verbose: boolean;

  constructor(options: AgentLoopOptions = {}) {
    this.maxIterations = options.maxIterations ?? 10;
    this.verbose = options.verbose ?? true;
  }

  /**
   * 发送消息给 Agent
   */
  async sendMessage(userInput: string): Promise<string> {
    // 添加用户消息
    this.messages.push(new HumanMessage(userInput));

    // 如果是第一轮，添加系统提示词
    const hasSystemMessage = this.messages.some(m => m instanceof SystemMessage);
    let allMessages = this.messages;
    if (!hasSystemMessage) {
      allMessages = [new SystemMessage(SYSTEM_PROMPT), ...this.messages];
    }

    let iterationCount = 0;

    if (this.verbose) {
      console.log("\n" + "=".repeat(50));
      console.log("用户：", userInput);
      console.log("=".repeat(50));
    }

    // Agent Loop
    while (iterationCount < this.maxIterations) {
      iterationCount++;

      if (this.verbose) {
        console.log(`\n[迭代 ${iterationCount}]`);
      }

      // Agent 思考
      const aiMessage = await modelWithTools.invoke(allMessages);
      allMessages.push(aiMessage);
      this.messages.push(aiMessage);

      // 检查是否需要调用工具
      if (isAIMessage(aiMessage) && aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
        if (this.verbose) {
          console.log(`Agent 调用 ${aiMessage.tool_calls.length} 个工具：`);
        }

        // 执行工具
        for (const toolCall of aiMessage.tool_calls) {
          if (this.verbose) {
            console.log(`  → ${toolCall.name}: ${JSON.stringify(toolCall.args)}`);
          }

          const toolFunc = toolsByName[toolCall.name];
          let result: string;

          if (toolFunc) {
            result = await toolFunc(toolCall.args);
          } else {
            result = `错误：未知工具 "${toolCall.name}"`;
          }

          if (this.verbose) {
            console.log(`    结果: ${result}`);
          }

          // 添加工具结果
          const toolMessage = new ToolMessage({
            content: result,
            tool_call_id: toolCall.id ?? "unknown",
          });
          allMessages.push(toolMessage);
          this.messages.push(toolMessage);
        }
      } else {
        // 最终答案
        const finalAnswer = String(aiMessage.content);

        if (this.verbose) {
          console.log("\nAI：", finalAnswer);
          console.log("-".repeat(50));
        }

        return finalAnswer;
      }
    }

    // 达到最大迭代次数
    return "达到最大迭代次数，无法给出答案";
  }

  /**
   * 获取对话历史
   */
  getHistory(): BaseMessage[] {
    return [...this.messages];
  }

  /**
   * 清空对话历史
   */
  clearHistory(): void {
    this.messages = [];
  }
}

/**
 * ============================================
 * 第六部分：各种演示场景
 * ============================================
 */

/**
 * 演示 1：基础 Agent Loop
 * 展示 Agent 如何处理简单问题
 */
async function runBasicAgentDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 1：基础 Agent Loop");
  console.log("=".repeat(60));

  const testCases = [
    "北京今天天气怎么样？",
    "计算 123 + 456 等于多少？",
    "什么是 LangChain？",
  ];

  for (const testCase of testCases) {
    const result = await runAgentLoop(testCase, { maxIterations: 5 });
    console.log("\n等待 2 秒...\n");
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

/**
 * 演示 2：多步骤推理
 * 展示 Agent 如何组合多个工具完成复杂任务
 */
async function runMultiStepDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 2：多步骤推理");
  console.log("=".repeat(60));

  const complexQueries = [
    "北京和上海哪个城市温度更高？相差多少度？",
    "搜索 LangChain 的信息，然后计算 100 除以 4 的结果",
    "查询北京天气，如果温度超过 30 度，计算温度减去 5 的结果",
  ];

  for (const query of complexQueries) {
    const result = await runAgentLoop(query, { maxIterations: 8 });
    console.log("\n等待 3 秒...\n");
    await new Promise((resolve) => setTimeout(resolve, 3000));
  }
}

/**
 * 演示 3：带记忆的对话
 * 展示 Agent 如何在多轮对话中保持上下文
 */
async function runConversationalDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 3：带记忆的对话");
  console.log("=".repeat(60));

  const agent = new ConversationalAgent({ maxIterations: 10 });

  const conversation = [
    "你好，我想知道北京的天气。",
    "那儿的温度适合穿什么衣服？",
    "帮我计算一下，如果我有 500 元，买 3 件 100 元的衣服还剩多少钱？",
    "刚才我说的城市天气怎么样？",
  ];

  for (const userInput of conversation) {
    await agent.sendMessage(userInput);
    console.log("\n等待 2 秒...\n");
    await new Promise((resolve) => setTimeout(resolve, 2000));
  }
}

/**
 * 演示 4：迭代过程可视化
 * 展示 Agent 的完整思考过程
 */
async function runVisualizationDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 4：迭代过程可视化");
  console.log("=".repeat(60));

  const query = "搜索 DeepSeek 的信息，然后计算 2024 除以 4 的结果";

  console.log("\n本演示将展示 Agent 的完整思考过程：\n");
  console.log("1. 首先，Agent 会分析问题，决定调用哪些工具");
  console.log("2. 然后，执行工具并获取结果");
  console.log("3. 根据结果继续思考，可能需要调用更多工具");
  console.log("4. 最终给出综合答案\n");

  const result = await runAgentLoop(query, { maxIterations: 8, verbose: true });

  console.log("\n" + "=".repeat(60));
  console.log("执行摘要：");
  console.log(`- 总迭代次数：${result.iterationCount}`);
  console.log(`- 消息总数：${result.messages.length}`);
  console.log(`- 最终答案：${result.finalAnswer}`);
  console.log("=".repeat(60));
}

/**
 * 演示 5：最大迭代次数限制
 * 展示循环保护机制
 */
async function runMaxIterationDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 5：最大迭代次数限制");
  console.log("=".repeat(60));

  console.log("\n本演示设置最大迭代次数为 2，展示循环保护机制\n");

  // 设置一个很低的最大迭代次数，强制触发限制
  const result = await runAgentLoop("北京和上海哪个城市温度更高？相差多少度？", {
    maxIterations: 2,
    verbose: true,
  });

  console.log("\n由于设置了最大迭代次数为 2，Agent 可能无法完成所有工具调用");
  console.log("这在实际应用中可以防止无限循环或过度消耗资源");
}

/**
 * ============================================
 * 第七部分：主函数
 * ============================================
 */

/**
 * 主函数
 * 运行所有 Agent Loop 演示
 */
async function main() {
  console.log("LangChain Agent Loop (智能体循环) 功能演示");
  console.log("=".repeat(60));
  console.log("\n本演示展示如何构建智能体循环");
  console.log("Agent Loop 允许 AI 进行多轮思考和工具调用\n");

  try {
    // 运行基础 Agent Loop 演示
    await runBasicAgentDemo();

    // 运行多步骤推理演示
    await runMultiStepDemo();

    // 运行带记忆的对话演示
    await runConversationalDemo();

    // 运行迭代过程可视化演示
    await runVisualizationDemo();

    // 运行最大迭代次数限制演示
    await runMaxIterationDemo();

    console.log("\n" + "=".repeat(60));
    console.log("所有演示运行完成！");
    console.log("=".repeat(60));
  } catch (error) {
    console.error("运行出错：", error);
  }
}

// 启动程序
main();
