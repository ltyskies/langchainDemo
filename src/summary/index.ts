import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate, MessagesPlaceholder } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { RunnableSequence } from "@langchain/core/runnables";
import { AIMessage, HumanMessage } from "@langchain/core/messages";
import dotenv from "dotenv";

// 加载环境变量，从 .env 文件中读取配置
dotenv.config();

/**
 * 初始化 ChatOpenAI 模型实例
 * 这里使用 DeepSeek API，通过配置 baseURL 来实现
 */
const model = new ChatOpenAI({
  model: process.env.DEEPSEEK_MODEL,        // 模型名称，如 "deepseek-chat"
  apiKey: process.env.DEEPSEEK_API_KEY,     // DeepSeek API 密钥
  configuration: {
    baseURL: process.env.DEEPSEEK_BASE_URL, // DeepSeek API 基础地址
  },
  temperature: 0.7, // 温度参数，控制生成文本的随机性，值越大越随机
});

// 创建字符串输出解析器，将模型的输出转换为纯文本字符串
const outputParser = new StringOutputParser();

/**
 * 简单记忆类 - 用于存储和管理对话历史
 * 这是 LangChain 记忆功能的核心概念，通过保存历史消息让 AI 记住上下文
 */
class SimpleMemory {
  // 存储消息数组，包含人类消息和 AI 消息
  private messages: Array<HumanMessage | AIMessage> = [];
  // 最大保留的对话轮数（一轮 = 人类消息 + AI 消息）
  private maxMessages: number;

  /**
   * 构造函数
   * @param maxMessages - 最大保留的对话轮数，默认为 10 轮
   */
  constructor(maxMessages: number = 10) {
    this.maxMessages = maxMessages;
  }

  /**
   * 加载记忆变量
   * 在每次调用链时，会将保存的历史消息注入到提示词模板中
   * @returns 包含历史消息的对象
   */
  async loadMemoryVariables(): Promise<{ history: Array<HumanMessage | AIMessage> }> {
    return { history: this.messages };
  }

  /**
   * 保存对话上下文
   * 将用户输入和 AI 回复保存到记忆中
   * @param input - 用户的输入文本
   * @param output - AI 的回复文本
   */
  async saveContext(input: string, output: string): Promise<void> {
    // 将用户输入封装为 HumanMessage 并添加到消息列表
    this.messages.push(new HumanMessage(input));
    // 将 AI 回复封装为 AIMessage 并添加到消息列表
    this.messages.push(new AIMessage(output));

    // 如果消息数量超过限制，只保留最近的消息
    // maxMessages * 2 是因为每轮对话包含两条消息（人类 + AI）
    if (this.messages.length > this.maxMessages * 2) {
      this.messages = this.messages.slice(-this.maxMessages * 2);
    }
  }

  /**
   * 清空所有记忆
   * 用于开始新的对话会话
   */
  clear(): void {
    this.messages = [];
  }
}

/**
 * 基础记忆功能演示
 * 展示如何使用 SimpleMemory 让 AI 记住用户信息和对话历史
 */
async function runBasicMemoryDemo() {
  // 创建一个记忆实例，使用默认的 10 轮对话限制
  const memory = new SimpleMemory();

  /**
   * 创建提示词模板
   * 包含三个部分：
   * 1. system - 系统提示词，定义 AI 的角色和行为
   * 2. MessagesPlaceholder("history") - 历史消息占位符，会被实际的历史消息替换
   * 3. human - 用户的当前输入
   */
  const promptTemplate = ChatPromptTemplate.fromMessages([
    ["system", "你是一个友好的AI助手，记住对话历史来回答用户问题。"],
    new MessagesPlaceholder("history"), // 这里会被替换为保存的历史消息
    ["human", "{input}"],
  ]);

  /**
   * 构建处理链（Chain）
   * RunnableSequence.from 按顺序执行以下步骤：
   * 1. 准备输入数据（input 和 history）
   * 2. 将数据传入提示词模板生成完整提示
   * 3. 调用语言模型生成回复
   * 4. 解析输出为字符串
   */
  const chain = RunnableSequence.from([
    {
      // 提取用户输入
      input: (input: { input: string }) => input.input,
      // 从记忆中加载历史消息
      history: async () => {
        const memoryResult = await memory.loadMemoryVariables();
        return memoryResult.history;
      },
    },
    promptTemplate, // 提示词模板
    model,          // 语言模型
    outputParser,   // 输出解析器
  ]);

  console.log("=== 基础记忆功能演示 ===\n");

  // 定义测试对话序列
  const inputs = [
    "你好，我叫小明。",      // 第一轮：自我介绍
    "我叫什么名字？",        // 第二轮：测试记忆（应该记得叫小明）
    "我喜欢吃披萨。",        // 第三轮：分享喜好
    "我喜欢吃什么？",        // 第四轮：测试记忆（应该记得喜欢吃披萨）
  ];

  // 依次执行对话
  for (const input of inputs) {
    console.log(`用户: ${input}`);
    // 调用链获取 AI 回复
    const response = await chain.invoke({ input });
    console.log(`AI: ${response}\n`);

    // 将本轮对话保存到记忆中，供下一轮使用
    await memory.saveContext(input, response);
  }
}

/**
 * 旅行顾问记忆演示
 * 展示记忆功能在实际场景中的应用 - 旅行规划
 * AI 会根据用户逐步提供的信息（目的地、喜好、预算）给出个性化建议
 */
async function runConversationWithMemoryDemo() {
  const memory = new SimpleMemory();

  const promptTemplate = ChatPromptTemplate.fromMessages([
    [
      "system",
      "你是一位专业的旅行顾问。根据用户的对话历史，提供个性化的旅行建议。",
    ],
    new MessagesPlaceholder("history"),
    ["human", "{input}"],
  ]);

  const chain = RunnableSequence.from([
    {
      input: (input: { input: string }) => input.input,
      history: async () => {
        const memoryResult = await memory.loadMemoryVariables();
        return memoryResult.history;
      },
    },
    promptTemplate,
    model,
    outputParser,
  ]);

  console.log("\n=== 旅行顾问记忆演示 ===\n");

  // 模拟用户逐步提供旅行需求信息
  const conversations = [
    "我想去日本旅游。",                    // 第 1 轮：确定目的地
    "我喜欢历史文化和美食。",              // 第 2 轮：说明兴趣
    "预算大概1万人民币。",                 // 第 3 轮：说明预算
    "根据我的喜好和预算，推荐一些行程。",  // 第 4 轮：请求综合建议（需要记住前面的所有信息）
  ];

  for (const content of conversations) {
    console.log(`用户: ${content}`);
    const response = await chain.invoke({ input: content });
    console.log(`AI: ${response}\n`);

    // 保存每轮对话，让 AI 记住用户的完整需求
    await memory.saveContext(content, response);
  }
}

/**
 * 窗口记忆演示
 * 展示如何限制记忆长度，只保留最近的几轮对话
 * 这在长对话中很有用，可以避免上下文过长导致的问题
 */
async function runMemoryWindowDemo() {
  // 创建一个只保留最近 2 轮对话的记忆实例
  const memory = new SimpleMemory(2);

  const promptTemplate = ChatPromptTemplate.fromMessages([
    ["system", "你是一个问答助手，只根据最近的对话历史回答问题。"],
    new MessagesPlaceholder("history"),
    ["human", "{input}"],
  ]);

  const chain = RunnableSequence.from([
    {
      input: (input: { input: string }) => input.input,
      history: async () => {
        const memoryResult = await memory.loadMemoryVariables();
        return memoryResult.history;
      },
    },
    promptTemplate,
    model,
    outputParser,
  ]);

  console.log("\n=== 窗口记忆演示（只保留最近2轮对话）===\n");

  const inputs = [
    "我的幸运数字是7。",      // 第 1 轮：较早的信息
    "我喜欢蓝色。",            // 第 2 轮
    "我最喜欢的季节是秋天。",  // 第 3 轮（此时第 1 轮可能被移出记忆）
    "我的幸运数字是多少？",    // 测试：AI 可能不记得是 7，因为只保留了最近 2 轮
  ];

  for (const input of inputs) {
    console.log(`用户: ${input}`);
    const response = await chain.invoke({ input });
    console.log(`AI: ${response}\n`);

    await memory.saveContext(input, response);
  }
}

/**
 * 主函数
 * 依次运行三个记忆功能演示
 */
async function main() {
  try {
    // 运行基础记忆演示
    await runBasicMemoryDemo();
    // 运行旅行顾问场景演示
    await runConversationWithMemoryDemo();
    // 运行窗口记忆演示
    await runMemoryWindowDemo();
  } catch (error) {
    console.error("运行出错：", error);
  }
}

// 启动程序
main();
