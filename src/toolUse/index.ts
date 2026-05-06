import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { AIMessage, HumanMessage, ToolMessage } from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import dotenv from "dotenv";

// 加载环境变量，从 .env 文件中读取配置
dotenv.config();

/**
 * ============================================
 * 第一部分：工具（Tools）定义
 * ============================================
 * 
 * 工具是 LangChain 中让 AI 与外部世界交互的核心机制。
 * AI 可以根据用户输入决定调用哪个工具，并传入合适的参数。
 * 工具执行后，结果会返回给 AI，AI 再基于结果生成最终回答。
 */

/**
 * 工具 1：天气查询工具
 * 模拟查询指定城市的天气信息
 * 使用 Zod 定义参数结构，确保 AI 传入正确的参数
 */
const weatherTool = tool(
  async ({ city }: { city: string }) => {
    // 模拟天气数据（实际应用中应调用真实天气 API）
    const weatherData: Record<string, { temperature: number; condition: string; humidity: number }> = {
      "北京": { temperature: 25, condition: "晴朗", humidity: 45 },
      "上海": { temperature: 28, condition: "多云", humidity: 65 },
      "广州": { temperature: 32, condition: "雷阵雨", humidity: 80 },
      "深圳": { temperature: 31, condition: "阴天", humidity: 75 },
      "杭州": { temperature: 26, condition: "小雨", humidity: 70 },
    };

    const weather = weatherData[city];
    if (!weather) {
      return `抱歉，暂无 ${city} 的天气信息。目前支持查询：北京、上海、广州、深圳、杭州。`;
    }

    return `${city}当前天气：${weather.condition}，温度 ${weather.temperature}°C，湿度 ${weather.humidity}%。`;
  },
  {
    name: "get_weather", // 工具名称，AI 会通过这个名称调用工具
    description: "查询指定城市的当前天气信息，包括温度、天气状况和湿度", // 工具描述，帮助 AI 理解何时使用该工具
    schema: z.object({
      city: z.string().describe("要查询天气的城市名称，如：北京、上海、广州"), // 参数定义和描述
    }),
  }
);

/**
 * 工具 2：计算器工具
 * 执行基础数学运算
 */
const calculatorTool = tool(
  async ({ expression }: { expression: string }) => {
    try {
      // 安全地计算数学表达式
      // 只允许数字和基本运算符
      const sanitizedExpression = expression.replace(/[^0-9+\-*/.()\s]/g, "");
      
      // 使用 Function 构造函数安全地计算表达式
      const result = new Function(`return ${sanitizedExpression}`)();
      
      return `计算结果：${expression} = ${result}`;
    } catch (error) {
      return `计算出错：无法计算表达式 "${expression}"，请检查输入是否合法。`;
    }
  },
  {
    name: "calculator",
    description: "执行基础数学运算，支持加减乘除和括号，如：2 + 3 * 4、(100 - 50) / 5",
    schema: z.object({
      expression: z.string().describe("数学表达式，如：2 + 3、100 / 4、(50 - 20) * 3"),
    }),
  }
);

/**
 * 工具 3：时间查询工具
 * 获取当前时间信息
 */
const timeTool = tool(
  async ({ timezone }: { timezone?: string }) => {
    const now = new Date();
    const timeString = now.toLocaleString("zh-CN", {
      timeZone: timezone || "Asia/Shanghai",
      year: "numeric",
      month: "long",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      weekday: "long",
    });
    return `当前时间：${timeString}`;
  },
  {
    name: "get_current_time",
    description: "获取当前的日期和时间信息",
    schema: z.object({
      timezone: z.string().optional().describe("时区，如：Asia/Shanghai、America/New_York，默认为北京时间"),
    }),
  }
);

/**
 * 工具 4：翻译工具
 * 将文本翻译成指定语言
 */
const translateTool = tool(
  async ({ text, targetLanguage }: { text: string; targetLanguage: string }) => {
    // 模拟翻译结果（实际应用中应调用真实翻译 API）
    const translations: Record<string, Record<string, string>> = {
      "你好": {
        "英语": "Hello",
        "日语": "こんにちは",
        "法语": "Bonjour",
        "西班牙语": "Hola",
      },
      "谢谢": {
        "英语": "Thank you",
        "日语": "ありがとう",
        "法语": "Merci",
        "西班牙语": "Gracias",
      },
    };

    const translation = translations[text]?.[targetLanguage];
    if (translation) {
      return `"${text}" 翻译成${targetLanguage}是："${translation}"`;
    }

    // 对于未预定义的文本，返回模拟翻译
    return `"${text}" 翻译成${targetLanguage}：[模拟翻译结果] ${text} (${targetLanguage})`;
  },
  {
    name: "translate",
    description: "将文本翻译成指定的目标语言",
    schema: z.object({
      text: z.string().describe("要翻译的文本内容"),
      targetLanguage: z.string().describe("目标语言，如：英语、日语、法语、西班牙语"),
    }),
  }
);

/**
 * ============================================
 * 第二部分：模型初始化
 * ============================================
 */

/**
 * 初始化 ChatOpenAI 模型实例
 * 这里使用 DeepSeek API，通过配置 baseURL 来实现
 * 关键：需要绑定工具（bindTools），让模型知道可以使用哪些工具
 */
const model = new ChatOpenAI({
  model: process.env.DEEPSEEK_MODEL,        // 模型名称，如 "deepseek-chat"
  apiKey: process.env.DEEPSEEK_API_KEY,     // DeepSeek API 密钥
  configuration: {
    baseURL: process.env.DEEPSEEK_BASE_URL, // DeepSeek API 基础地址
  },
  temperature: 0.7, // 温度参数，控制生成文本的随机性
  maxRetries: 2,    // 失败时的重试次数
});

// 将工具绑定到模型上，这样模型就知道可以使用哪些工具
// 当用户的问题需要工具时，模型会输出 tool_calls 而不是直接回答
const modelWithTools = model.bindTools([
  weatherTool,
  calculatorTool,
  timeTool,
  translateTool,
]);

// 创建字符串输出解析器
const outputParser = new StringOutputParser();

/**
 * ============================================
 * 第三部分：工具调用核心逻辑
 * ============================================
 */

/**
 * 处理工具调用的核心函数
 * 这是 Tool Use 的核心流程：
 * 1. 用户输入 -> 模型判断是否需要调用工具
 * 2. 如果需要，模型输出 tool_calls（包含工具名和参数）
 * 3. 执行对应的工具函数
 * 4. 将工具执行结果返回给模型
 * 5. 模型基于工具结果生成最终回答
 * 
 * @param userInput 用户输入的文本
 * @returns AI 的最终回答
 */
async function processWithTools(userInput: string): Promise<string> {
  // 第一步：创建用户消息
  const messages = [new HumanMessage(userInput)];

  // 第二步：调用绑定了工具的模型
  // 模型会分析用户输入，决定是否需要调用工具
  const aiMessage = await modelWithTools.invoke(messages);

  // 第三步：检查模型是否请求了工具调用
  // tool_calls 是模型决定调用的工具列表
  if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
    console.log(`  [工具调用] 模型决定调用 ${aiMessage.tool_calls.length} 个工具`);

    // 将 AI 的消息（包含 tool_calls）添加到消息列表
    messages.push(aiMessage);

    // 第四步：依次执行每个工具调用
    for (const toolCall of aiMessage.tool_calls) {
      console.log(`    - 工具：${toolCall.name}，参数：${JSON.stringify(toolCall.args)}`);

      let toolResult: string;

      // 根据工具名称调用对应的工具函数
      switch (toolCall.name) {
        case "get_weather":
          toolResult = await weatherTool.invoke(toolCall.args);
          break;
        case "calculator":
          toolResult = await calculatorTool.invoke(toolCall.args);
          break;
        case "get_current_time":
          toolResult = await timeTool.invoke(toolCall.args);
          break;
        case "translate":
          toolResult = await translateTool.invoke(toolCall.args);
          break;
        default:
          toolResult = `未知工具：${toolCall.name}`;
      }

      console.log(`    - 工具返回：${toolResult}`);

      // 第五步：将工具执行结果封装为 ToolMessage 添加到消息列表
      // tool_call_id 用于关联工具调用和结果
      messages.push(
        new ToolMessage({
          content: toolResult,
          tool_call_id: toolCall.id || "unknown",
        })
      );
    }

    // 第六步：再次调用模型，传入工具执行结果
    // 模型会基于工具结果生成最终的自然语言回答
    const finalResponse = await modelWithTools.invoke(messages);
    return finalResponse.content as string;
  }

  // 如果没有工具调用，直接返回模型的回答
  return aiMessage.content as string;
}

/**
 * ============================================
 * 第四部分：各种演示场景
 * ============================================
 */

/**
 * 演示 1：基础工具调用
 * 展示单个工具的调用流程
 */
async function runBasicToolDemo() {
  console.log("=== 基础工具调用演示 ===\n");

  const testCases = [
    { input: "北京今天天气怎么样？", expectedTool: "天气查询" },
    { input: "计算 123 + 456 等于多少？", expectedTool: "计算器" },
    { input: "现在几点了？", expectedTool: "时间查询" },
    { input: "把\"你好\"翻译成英语", expectedTool: "翻译" },
  ];

  for (const testCase of testCases) {
    console.log(`用户：${testCase.input}`);
    console.log(`[预期调用：${testCase.expectedTool}工具]`);

    const response = await processWithTools(testCase.input);
    console.log(`AI：${response}\n`);
  }
}

/**
 * 演示 2：多工具组合调用
 * 展示 AI 如何在一个对话中调用多个工具
 */
async function runMultiToolDemo() {
  console.log("\n=== 多工具组合调用演示 ===\n");

  const complexQueries = [
    "上海和广州的天气怎么样？顺便告诉我现在几点了。",
    "计算 (100 + 200) * 3，然后把结果翻译成英语。",
  ];

  for (const query of complexQueries) {
    console.log(`用户：${query}`);
    const response = await processWithTools(query);
    console.log(`AI：${response}\n`);
  }
}

/**
 * 演示 3：带记忆的对话式工具调用
 * 展示如何在多轮对话中使用工具
 */
async function runConversationalToolDemo() {
  console.log("\n=== 对话式工具调用演示 ===\n");

  // 存储对话历史
  const messages: Array<HumanMessage | AIMessage | ToolMessage> = [];

  // 定义对话流程
  const conversation = [
    "你好，我想知道北京的天气。",
    "那儿的温度适合穿什么衣服？",
    "帮我计算一下，如果我有500元，买3件100元的衣服还剩多少钱？",
    "把剩下的钱数翻译成英语。",
  ];

  for (const userInput of conversation) {
    console.log(`用户：${userInput}`);

    // 添加用户消息到历史
    messages.push(new HumanMessage(userInput));

    // 调用模型（带工具）
    const aiMessage = await modelWithTools.invoke(messages);

    // 检查是否需要工具调用
    if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
      messages.push(aiMessage);

      // 执行工具调用
      for (const toolCall of aiMessage.tool_calls) {
        let toolResult: string;

        switch (toolCall.name) {
          case "get_weather":
            toolResult = await weatherTool.invoke(toolCall.args);
            break;
          case "calculator":
            toolResult = await calculatorTool.invoke(toolCall.args);
            break;
          case "translate":
            toolResult = await translateTool.invoke(toolCall.args);
            break;
          default:
            toolResult = `未知工具：${toolCall.name}`;
        }

        messages.push(
          new ToolMessage({
            content: toolResult,
            tool_call_id: toolCall.id || "unknown",
          })
        );
      }

      // 获取最终回答
      const finalResponse = await modelWithTools.invoke(messages);
      console.log(`AI：${finalResponse.content}\n`);
      messages.push(new AIMessage(finalResponse.content as string));
    } else {
      console.log(`AI：${aiMessage.content}\n`);
      messages.push(new AIMessage(aiMessage.content as string));
    }
  }
}

/**
 * 演示 4：工具调用链
 * 使用 LangChain 的 RunnableSequence 构建更复杂的工具调用流程
 */
async function runToolChainDemo() {
  console.log("\n=== 工具调用链演示 ===\n");

  /**
   * 构建一个工具调用链
   * 这个链会：
   * 1. 接收用户输入
   * 2. 调用模型判断是否需要工具
   * 3. 如果需要，执行工具并获取结果
   * 4. 生成最终回答
   */
  const toolChain = RunnableSequence.from([
    {
      // 准备输入
      input: (input: { question: string }) => input.question,
    },
    async (input: { input: string }) => {
      const messages = [new HumanMessage(input.input)];
      const aiMessage = await modelWithTools.invoke(messages);

      // 处理工具调用
      if (aiMessage.tool_calls && aiMessage.tool_calls.length > 0) {
        messages.push(aiMessage);

        for (const toolCall of aiMessage.tool_calls) {
          let toolResult: string;

          switch (toolCall.name) {
            case "get_weather":
              toolResult = await weatherTool.invoke(toolCall.args);
              break;
            case "calculator":
              toolResult = await calculatorTool.invoke(toolCall.args);
              break;
            case "get_current_time":
              toolResult = await timeTool.invoke(toolCall.args);
              break;
            case "translate":
              toolResult = await translateTool.invoke(toolCall.args);
              break;
            default:
              toolResult = `未知工具：${toolCall.name}`;
          }

          messages.push(
            new ToolMessage({
              content: toolResult,
              tool_call_id: toolCall.id || "unknown",
            })
          );
        }

        const finalResponse = await modelWithTools.invoke(messages);
        return finalResponse.content;
      }

      return aiMessage.content;
    },
  ]);

  // 测试链
  const questions = [
    "深圳的天气如何？",
    "计算 999 / 3 + 100",
    "现在是什么时间？",
  ];

  for (const question of questions) {
    console.log(`用户：${question}`);
    const response = await toolChain.invoke({ question });
    console.log(`AI：${response}\n`);
  }
}

/**
 * 演示 5：工具调用错误处理
 * 展示如何处理工具调用中的错误情况
 */
async function runErrorHandlingDemo() {
  console.log("\n=== 工具调用错误处理演示 ===\n");

  const errorCases = [
    "查询火星的天气", // 不存在的城市
    "计算 abc + 123", // 非法的数学表达式
    "翻译一段超长文本到所有已知的语言", // 边界情况
  ];

  for (const input of errorCases) {
    console.log(`用户：${input}`);
    try {
      const response = await processWithTools(input);
      console.log(`AI：${response}\n`);
    } catch (error) {
      console.log(`错误：${error}\n`);
    }
  }
}

/**
 * ============================================
 * 第五部分：主函数
 * ============================================
 */

/**
 * 主函数
 * 依次运行所有工具调用演示
 */
async function main() {
  try {
    console.log("LangChain Tool Use 功能演示\n");
    console.log("============================\n");

    // 运行基础工具调用演示
    await runBasicToolDemo();

    // 运行多工具组合调用演示
    await runMultiToolDemo();

    // 运行对话式工具调用演示
    await runConversationalToolDemo();

    // 运行工具调用链演示
    await runToolChainDemo();

    // 运行错误处理演示
    await runErrorHandlingDemo();

    console.log("\n所有演示运行完成！");
  } catch (error) {
    console.error("运行出错：", error);
  }
}

// 启动程序
main();
