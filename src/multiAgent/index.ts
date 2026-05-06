/**
 * ============================================
 * LangChain Multi-Agent (多智能体协作) 功能演示
 * ============================================
 *
 * 本演示展示了如何使用 LangChain + LangGraph 构建多智能体协作系统。
 * 多智能体系统（Multi-Agent System）是指多个具有不同角色和能力的 AI 智能体
 * 协同工作，共同完成复杂任务的系统架构。
 *
 * 本 demo 包含以下内容：
 *   1. 多智能体协作核心概念介绍
 *   2. LangGraph StateGraph 状态图的使用
 *   3. Supervisor（调度员）模式的实现
 *   4. 条件路由（Conditional Routing）机制
 *   5. 智能体间消息传递与状态共享
 *   6. 工具调用在多智能体中的集成
 *   7. 流式输出与执行过程可视化
 *
 * 核心架构 —— Supervisor 模式：
 *   ┌──────────────────────────────────────────────┐
 *   │                Supervisor（调度员）             │
 *   │       分析任务 → 分配给合适的智能体 → 汇总结果    │
 *   └───────┬────────────┬────────────┬────────────┘
 *           │            │            │
 *           ▼            ▼            ▼
 *     ┌──────────┐ ┌──────────┐ ┌──────────┐
 *     │Researcher │ │  Writer  │ │ Reviewer  │
 *     │ (研究员)  │ │ (写作者) │ │ (审核员)  │
 *     │ 搜索信息  │ │ 撰写内容 │ │ 审核反馈  │
 *     └──────────┘ └──────────┘ └──────────┘
 *           │            │            │
 *           └────────────┴────────────┘
 *                        │
 *                        ▼
 *                  返回 Supervisor
 *
 * 工作流程：
 *   1. 用户输入 → Supervisor 分析任务
 *   2. Supervisor 通过 route_to_agent 工具决定下一个智能体
 *   3. 智能体执行任务，结果写入共享状态
 *   4. Supervisor 根据最新状态决定下一步
 *   5. 重复 2-4，直到 Supervisor 判断任务完成（FINISH）
 *
 * LangGraph 关键概念：
 *   - StateGraph：有状态的图，节点之间通过共享状态通信
 *   - Annotation：定义状态的结构和更新方式（reducer）
 *   - Node（节点）：图中的处理单元，对应一个智能体
 *   - Edge（边）：节点之间的连接，定义执行顺序
 *   - Conditional Edge（条件边）：根据状态动态选择下一个节点
 *   - START / END：特殊的图入口和出口节点
 *
 * 前置要求：
 *   - 安装依赖：pnpm add @langchain/langgraph @langchain/openai @langchain/core zod
 *   - 配置环境变量：DEEPSEEK_API_KEY, DEEPSEEK_BASE_URL, DEEPSEEK_MODEL
 */

import { ChatOpenAI } from "@langchain/openai";
import {
  StateGraph,
  Annotation,
  START,
  END,
  messagesStateReducer,
  GraphRecursionError,
} from "@langchain/langgraph";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  ToolMessage,
  BaseMessage,
  isAIMessage,
} from "@langchain/core/messages";
import dotenv from "dotenv";

dotenv.config();

/**
 * ============================================
 * 第一部分：多智能体协作核心概念
 * ============================================
 *
 * 什么是多智能体协作？
 * --------------------
 * 多智能体协作是指多个 AI 智能体按照一定的工作流程协同完成任务的机制。
 * 每个智能体有自己的角色、能力和工具，通过共享状态进行通信。
 *
 * 为什么需要多智能体？
 * --------------------
 * 1. 专业化分工：每个智能体专注于自己擅长的领域，提高输出质量
 * 2. 复杂任务分解：将复杂任务拆分为多个子任务，分别处理
 * 3. 质量控制：通过审核机制确保输出质量
 * 4. 灵活路由：根据任务类型动态选择最合适的智能体
 *
 * 常见的多智能体架构：
 * --------------------
 * 1. Supervisor 模式：一个调度员协调所有智能体（本 demo 采用）
 * 2. 顺序流水线：智能体按固定顺序依次处理
 * 3. 层级模式：多级调度，大 Supervisor 管理小 Supervisor
 * 4. 对等模式：智能体之间直接通信，无中心调度
 *
 * LangGraph 的角色：
 * --------------------
 * LangGraph 是 LangChain 生态中用于构建有状态、多参与者应用的框架。
 * 它提供了 StateGraph 来定义节点（智能体）和边（流转逻辑），
 * 支持条件路由、循环、状态持久化等高级特性。
 */

/**
 * ============================================
 * 第二部分：工具定义
 * ============================================
 *
 * 在多智能体系统中，不同的智能体可以拥有不同的工具集。
 * 这里定义了研究员使用的搜索工具和天气工具，
 * 以及调度员使用的路由工具。
 */

/**
 * 工具 1：搜索引擎工具
 * 供 Researcher（研究员）使用，用于搜索互联网获取信息
 */
const searchTool = tool(
  async ({ query }: { query: string }) => {
    const searchResults: Record<string, string> = {
      LangChain:
        "LangChain 是一个用于构建 LLM 应用程序的框架，提供了链、代理、记忆等核心组件，支持 Python 和 JavaScript。",
      DeepSeek:
        "DeepSeek 是一家中国 AI 公司，开发了 DeepSeek-V3 等大语言模型，以高性价比和开源策略著称。",
      人工智能:
        "人工智能（AI）是计算机科学的一个分支，致力于创建能够模拟人类智能的系统，包括机器学习、深度学习、自然语言处理等子领域。",
      量子计算:
        "量子计算利用量子力学原理（如叠加和纠缠）进行计算，有望在密码学、药物发现、优化问题等领域超越经典计算机。",
      TypeScript:
        "TypeScript 是 JavaScript 的超集，添加了静态类型系统，由微软开发维护，广泛用于大型 Web 应用开发。",
    };

    for (const [key, value] of Object.entries(searchResults)) {
      if (query.includes(key)) {
        return `搜索结果：${value}`;
      }
    }
    return `搜索结果：找到关于 "${query}" 的相关信息。这是一个模拟搜索结果，实际应用中应调用真实搜索 API。`;
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
 * 工具 2：天气查询工具
 * 供 Researcher（研究员）使用，用于查询城市天气
 */
const weatherTool = tool(
  async ({ city }: { city: string }) => {
    const weatherData: Record<string, { temp: number; condition: string; humidity: number }> = {
      北京: { temp: 25, condition: "晴朗", humidity: 45 },
      上海: { temp: 28, condition: "多云", humidity: 65 },
      广州: { temp: 32, condition: "雷阵雨", humidity: 80 },
      深圳: { temp: 31, condition: "阴天", humidity: 75 },
      杭州: { temp: 26, condition: "小雨", humidity: 70 },
    };

    const weather = weatherData[city];
    if (weather) {
      return `${city}当前天气：${weather.condition}，温度 ${weather.temp}°C，湿度 ${weather.humidity}%`;
    }
    return `暂无 ${city} 的天气信息。目前支持查询：北京、上海、广州、深圳、杭州。`;
  },
  {
    name: "get_weather",
    description: "查询指定城市的当前天气信息，包括温度、天气状况和湿度",
    schema: z.object({
      city: z.string().describe("城市名称，如：北京、上海、广州"),
    }),
  }
);

/**
 * 工具 3：路由工具
 * 供 Supervisor（调度员）使用，用于决定将任务分配给哪个智能体
 *
 * 这是 Supervisor 模式的关键：调度员通过调用此工具来决定下一步路由，
 * 而不是直接输出文本。这确保了路由决策的结构化和可靠性。
 */
const routeTool = tool(
  async ({ next, reasoning }: { next: string; reasoning: string }) => {
    return JSON.stringify({ next, reasoning });
  },
  {
    name: "route_to_agent",
    description:
      "将任务路由到指定的智能体。你必须始终使用此工具来决定下一步。",
    schema: z.object({
      next: z
        .enum(["researcher", "writer", "reviewer", "FINISH"])
        .describe(
          "下一个要调用的智能体：researcher（研究员）、writer（写作者）、reviewer（审核员），或 FINISH 表示任务完成"
        ),
      reasoning: z.string().describe("选择该智能体或结束任务的原因"),
    }),
  }
);

/**
 * ============================================
 * 第三部分：模型初始化
 * ============================================
 *
 * 使用 DeepSeek API 作为 LLM 后端
 * 所有智能体共享同一个模型实例，但使用不同的系统提示词和工具绑定
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
 * ============================================
 * 第四部分：状态定义
 * ============================================
 *
 * 在 LangGraph 中，状态（State）是所有节点共享的数据结构。
 * 每个节点可以读取状态、修改状态，修改后的状态会传递给下一个节点。
 *
 * 状态字段的更新方式由 Annotation 定义：
 *   - 带 reducer 的字段：新值通过 reducer 函数与旧值合并（如 messages 的追加）
 *   - 不带 reducer 的字段：新值直接覆盖旧值（如 next 的更新）
 *
 * 本 demo 的状态包含：
 *   - messages：对话消息列表，使用 messagesStateReducer 追加合并
 *   - next：下一个要执行的智能体名称，直接覆盖
 */

const AgentState = Annotation.Root({
  /**
   * messages 字段：存储所有智能体的对话消息
   * 使用 messagesStateReducer 作为 reducer，它会：
   *   1. 将新消息追加到现有消息列表末尾
   *   2. 处理特殊的消息类型（如 RemoveMessage）
   * default: () => [] 表示初始状态为空数组
   */
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),

  /**
   * next 字段：存储 Supervisor 决定的下一个智能体名称
   * 不指定 reducer，默认使用 LastValue 策略（新值覆盖旧值）
   * 可能的值："researcher" | "writer" | "reviewer" | "FINISH"
   */
  next: Annotation<string>,
});

/**
 * ============================================
 * 第五部分：系统提示词
 * ============================================
 *
 * 每个智能体都有专属的系统提示词，定义其角色、职责和行为准则。
 * 这是实现智能体"专业化分工"的关键。
 */

/**
 * Supervisor（调度员）的系统提示词
 * 核心职责：分析任务 → 选择合适的智能体 → 汇总结果
 */
const SUPERVISOR_PROMPT = `你是一个任务调度员（Supervisor），负责分析用户任务并分配给合适的智能体。

你可以将任务分配给以下智能体：
- researcher（研究员）：擅长搜索信息、收集数据、调查事实。当需要获取外部信息时使用。
- writer（写作者）：擅长撰写文章、创作内容、整理文字。当需要生成文本内容时使用。
- reviewer（审核员）：擅长审核内容、提供反馈、质量检查。当需要检查输出质量时使用。

工作原则：
1. 仔细分析用户任务，判断需要哪些智能体参与
2. 如果需要先收集信息再写作，先分配给 researcher，再分配给 writer
3. 如果需要质量审核，在写作完成后分配给 reviewer
4. 如果任务简单，可以直接分配给一个智能体
5. 当所有工作完成后，选择 FINISH 结束任务

重要：你必须始终使用 route_to_agent 工具来做出路由决策，不要直接回复文本。`;

/**
 * Researcher（研究员）的系统提示词
 * 核心职责：搜索信息、收集数据、调查事实
 */
const RESEARCHER_PROMPT = `你是一个专业的研究员（Researcher），擅长搜索和收集信息。

你的职责：
1. 根据任务需求搜索相关信息
2. 整理和总结搜索结果
3. 提供准确、有价值的研究报告

你可以使用以下工具：
- search：搜索互联网获取信息
- get_weather：查询城市天气

工作原则：
- 优先使用工具获取一手信息
- 对搜索结果进行整理和归纳
- 提供清晰、有条理的研究报告
- 如果一次搜索不够，可以多次搜索`;

/**
 * Writer（写作者）的系统提示词
 * 核心职责：基于研究结果撰写内容
 */
const WRITER_PROMPT = `你是一个专业的写作者（Writer），擅长撰写各类文章和内容。

你的职责：
1. 根据提供的研究信息撰写文章
2. 确保内容清晰、有条理、引人入胜
3. 适当使用标题、段落等格式组织内容

写作原则：
- 内容准确，基于提供的研究信息
- 结构清晰，逻辑连贯
- 语言流畅，易于理解
- 适当使用标题和小标题组织内容
- 控制篇幅，突出重点`;

/**
 * Reviewer（审核员）的系统提示词
 * 核心职责：审核内容质量，提供改进建议
 */
const REVIEWER_PROMPT = `你是一个专业的审核员（Reviewer），擅长审核内容并提供反馈。

你的职责：
1. 审核内容的准确性和完整性
2. 检查逻辑和结构是否合理
3. 提供具体的改进建议

审核标准：
- 事实准确性：信息是否正确
- 逻辑连贯性：论述是否自洽
- 内容完整性：是否涵盖了关键要点
- 表达清晰度：语言是否通顺易懂

请给出详细的审核意见。如果内容质量良好，请明确指出优点。
如果需要改进，请给出具体的修改建议。`;

/**
 * ============================================
 * 第六部分：智能体节点函数
 * ============================================
 *
 * 在 LangGraph 中，每个节点是一个异步函数，接收当前状态作为参数，
 * 返回状态的更新（partial state）。
 *
 * 节点函数的执行流程：
 *   1. 从状态中读取需要的信息（如 messages）
 *   2. 构造提示词，调用 LLM
 *   3. 处理 LLM 的响应（可能包含工具调用）
 *   4. 返回状态更新
 */

/**
 * Supervisor 节点
 *
 * 调度员的核心逻辑：
 *   1. 读取当前对话消息
 *   2. 调用绑定了 route_to_agent 工具的 LLM
 *   3. LLM 通过工具调用决定下一个智能体
 *   4. 返回更新后的消息和 next 字段
 *
 * @param state - 当前图状态
 * @returns 状态更新（messages 和 next）
 */
async function supervisorNode(state: typeof AgentState.State) {
  console.log("\n📋 [Supervisor] 分析任务并决定路由...");

  /**
   * 将路由工具绑定到模型
   * 这样模型就会通过 tool_call 来表达路由决策
   * 而不是在文本中模糊地描述
   */
  const supervisorModel = model.bindTools([routeTool]);

  /**
   * 构造消息列表：系统提示词 + 当前对话历史
   * Supervisor 需要看到所有之前的消息来做出决策
   */
  const response = await supervisorModel.invoke([
    new SystemMessage(SUPERVISOR_PROMPT),
    ...state.messages,
  ]);

  /**
   * 解析模型的路由决策
   * 模型应该调用 route_to_agent 工具，包含 next 和 reasoning 参数
   */
  if (isAIMessage(response) && response.tool_calls && response.tool_calls.length > 0) {
    const toolCall = response.tool_calls[0];
    const next = toolCall.args.next as string;
    const reasoning = toolCall.args.reasoning as string;

    if (next === "FINISH") {
      console.log(`   ✅ 任务完成：${reasoning}`);
    } else {
      console.log(`   → 路由到：${next}`);
      console.log(`   → 原因：${reasoning}`);
    }

    return {
      messages: [
        new AIMessage(
          `[Supervisor] ${next === "FINISH" ? "任务完成" : `决定将任务交给 ${next}`}：${reasoning}`
        ),
      ],
      next,
    };
  }

  /**
   * 兜底处理：如果模型没有调用路由工具，直接结束
   * 这种情况理论上不应该发生（因为提示词明确要求使用工具）
   */
  console.log("   ⚠️ Supervisor 未调用路由工具，直接结束");
  return {
    messages: [response],
    next: "FINISH",
  };
}

/**
 * Researcher 节点
 *
 * 研究员的核心逻辑：
 *   1. 读取当前对话消息
 *   2. 调用绑定了搜索工具的 LLM
 *   3. 如果 LLM 请求工具调用，执行工具并获取结果
 *   4. 将工具结果反馈给 LLM，获取最终研究报告
 *   5. 返回更新后的消息
 *
 * 注意：研究员内部实现了 mini agent loop，
 * 支持多轮工具调用，直到获得足够的信息。
 *
 * @param state - 当前图状态
 * @returns 状态更新（messages 和 next）
 */
async function researcherNode(state: typeof AgentState.State) {
  console.log("\n🔍 [Researcher] 开始研究...");

  /**
   * 将搜索工具绑定到模型
   * 研究员可以使用 search 和 get_weather 工具
   */
  const researcherTools = [searchTool, weatherTool];
  const researcherModel = model.bindTools(researcherTools);

  /**
   * 构造消息列表：研究员系统提示词 + 当前对话历史
   */
  const messages: BaseMessage[] = [
    new SystemMessage(RESEARCHER_PROMPT),
    ...state.messages,
  ];

  /**
   * 调用模型，获取第一轮响应
   */
  let response = await researcherModel.invoke(messages);
  messages.push(response);

  /**
   * 处理工具调用循环
   * 研究员可能需要多次调用工具来收集足够的信息
   * 最多允许 3 轮工具调用，防止无限循环
   */
  let toolCallRounds = 0;
  const maxToolCallRounds = 3;

  while (
    isAIMessage(response) &&
    response.tool_calls &&
    response.tool_calls.length > 0 &&
    toolCallRounds < maxToolCallRounds
  ) {
    toolCallRounds++;

    for (const toolCall of response.tool_calls) {
      console.log(
        `   → 调用工具：${toolCall.name}(${JSON.stringify(toolCall.args)})`
      );

      /**
       * 根据工具名称执行对应的工具函数
       */
      let result: string;
      if (toolCall.name === "search") {
        result = String(await searchTool.invoke(toolCall.args as any));
      } else if (toolCall.name === "get_weather") {
        result = String(await weatherTool.invoke(toolCall.args as any));
      } else {
        result = `未知工具：${toolCall.name}`;
      }

      console.log(`   → 工具结果：${result}`);

      /**
       * 将工具结果封装为 ToolMessage，添加到消息列表
       * tool_call_id 用于关联工具调用和结果
       */
      messages.push(
        new ToolMessage({
          content: result,
          tool_call_id: toolCall.id ?? "unknown",
        })
      );
    }

    /**
     * 将工具结果反馈给模型，获取下一轮响应
     * 模型可能会继续调用工具，或者给出最终的研究报告
     */
    response = await researcherModel.invoke(messages);
    messages.push(response);
  }

  console.log("   ✅ 研究完成");

  /**
   * 返回状态更新
   * messages: 研究员的最终输出（只返回最后的 AI 消息，避免中间过程污染状态）
   * next: 设置为 "supervisor"，表示返回调度员
   */
  const finalContent = String(response.content);

  return {
    messages: [new AIMessage(`[Researcher] 研究报告：\n${finalContent}`)],
    next: "supervisor",
  };
}

/**
 * Writer 节点
 *
 * 写作者的核心逻辑：
 *   1. 读取当前对话消息（包含研究员的研究报告）
 *   2. 调用 LLM 生成文章内容
 *   3. 返回更新后的消息
 *
 * 写作者不需要工具，直接基于已有信息生成内容
 *
 * @param state - 当前图状态
 * @returns 状态更新（messages 和 next）
 */
async function writerNode(state: typeof AgentState.State) {
  console.log("\n✍️ [Writer] 开始写作...");

  /**
   * 构造消息列表：写作者系统提示词 + 当前对话历史
   * 对话历史中包含了研究员的研究报告，写作者基于此进行创作
   */
  const response = await model.invoke([
    new SystemMessage(WRITER_PROMPT),
    ...state.messages,
  ]);

  console.log("   ✅ 写作完成");

  return {
    messages: [new AIMessage(`[Writer] 文章内容：\n${response.content}`)],
    next: "supervisor",
  };
}

/**
 * Reviewer 节点
 *
 * 审核员的核心逻辑：
 *   1. 读取当前对话消息（包含写作者的文章）
 *   2. 调用 LLM 进行审核
 *   3. 返回审核意见
 *
 * 审核员不需要工具，直接基于已有内容进行审核
 *
 * @param state - 当前图状态
 * @returns 状态更新（messages 和 next）
 */
async function reviewerNode(state: typeof AgentState.State) {
  console.log("\n📝 [Reviewer] 开始审核...");

  /**
   * 构造消息列表：审核员系统提示词 + 当前对话历史
   * 对话历史中包含了写作者的文章，审核员基于此进行审核
   */
  const response = await model.invoke([
    new SystemMessage(REVIEWER_PROMPT),
    ...state.messages,
  ]);

  console.log("   ✅ 审核完成");

  return {
    messages: [new AIMessage(`[Reviewer] 审核意见：\n${response.content}`)],
    next: "supervisor",
  };
}

/**
 * ============================================
 * 第七部分：构建多智能体状态图
 * ============================================
 *
 * 使用 LangGraph 的 StateGraph 构建多智能体协作图。
 *
 * StateGraph 的构建步骤：
 *   1. 创建 StateGraph 实例，传入状态定义
 *   2. 添加节点（addNode）：每个节点对应一个智能体
 *   3. 添加边（addEdge）：定义节点之间的固定流转
 *   4. 添加条件边（addConditionalEdges）：根据状态动态选择下一个节点
 *   5. 编译图（compile）：生成可执行的图
 *
 * 本 demo 的图结构：
 *
 *   START ──→ supervisor ──→ (conditional) ──→ researcher ──→ supervisor
 *                  │                              writer   ──→ supervisor
 *                  │                              reviewer  ──→ supervisor
 *                  └──→ (FINISH) ──→ END
 */

/**
 * 创建多智能体协作图
 *
 * @returns 编译后的可执行图
 */
function createMultiAgentGraph() {
  /**
   * 使用链式调用构建 StateGraph
   *
   * 注意：LangGraph 的 StateGraph 使用 TypeScript 的类型推断来追踪节点名称。
   * addNode() 返回新的 StateGraph 类型（扩展了节点名称联合类型），
   * 因此必须使用链式调用来确保类型正确推断。
   *
   * 如果将 addNode() 和 addEdge() 分开调用，
   * TypeScript 无法知道后续的边引用了已添加的节点名称。
   *
   * 图结构：
   *   START → supervisor → (conditional) → researcher → supervisor
   *                                  → writer    → supervisor
   *                                  → reviewer  → supervisor
   *                                  → FINISH → END
   */
  return new StateGraph(AgentState)
    .addNode("supervisor", supervisorNode)
    .addNode("researcher", researcherNode)
    .addNode("writer", writerNode)
    .addNode("reviewer", reviewerNode)
    .addEdge(START, "supervisor")
    .addConditionalEdges("supervisor", (state) => state.next, {
      researcher: "researcher",
      writer: "writer",
      reviewer: "reviewer",
      FINISH: END,
    })
    .addEdge("researcher", "supervisor")
    .addEdge("writer", "supervisor")
    .addEdge("reviewer", "supervisor")
    .compile();
}

/**
 * ============================================
 * 第八部分：各种演示场景
 * ============================================
 */

/**
 * 演示 1：基础路由 —— 单智能体任务
 *
 * 展示 Supervisor 如何将简单任务路由到单个智能体
 * 场景：用户只需要搜索信息，Supervisor 将任务分配给 Researcher
 */
async function runBasicRoutingDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 1：基础路由 —— 单智能体任务");
  console.log("=".repeat(60));
  console.log("\n场景：用户需要搜索信息，Supervisor 分配给 Researcher\n");

  const graph = createMultiAgentGraph();

  try {
    /**
     * 调用图执行
     * invoke() 方法会同步执行整个图，直到到达 END 节点
     *
     * 参数说明：
     *   - 第一个参数：初始状态，包含用户消息
     *   - recursionLimit：最大递归次数，防止无限循环
     */
    const result = await graph.invoke(
      {
        messages: [new HumanMessage("帮我搜索一下 LangChain 是什么？")],
      },
      { recursionLimit: 10 }
    );

    console.log("\n" + "-".repeat(40));
    console.log("📊 执行结果摘要：");
    console.log(`   总消息数：${result.messages.length}`);
    const lastMessage = result.messages[result.messages.length - 1];
    console.log(
      `   最终输出：${String(lastMessage.content).substring(0, 200)}${String(lastMessage.content).length > 200 ? "..." : ""}`
    );
  } catch (error) {
    if (error instanceof GraphRecursionError) {
      console.error("❌ 达到最大递归次数，图执行被终止");
    } else {
      console.error("❌ 执行出错：", error);
    }
  }
}

/**
 * 演示 2：双智能体协作 —— 研究 + 写作
 *
 * 展示两个智能体如何协作完成一个需要多步骤的任务
 * 场景：先由 Researcher 搜索信息，再由 Writer 基于研究结果撰写文章
 */
async function runResearchAndWriteDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 2：双智能体协作 —— 研究 + 写作");
  console.log("=".repeat(60));
  console.log("\n场景：先研究量子计算，再撰写科普文章\n");

  const graph = createMultiAgentGraph();

  try {
    const result = await graph.invoke(
      {
        messages: [
          new HumanMessage(
            "请先研究量子计算的基本概念，然后写一篇简短的科普文章介绍量子计算。"
          ),
        ],
      },
      { recursionLimit: 15 }
    );

    console.log("\n" + "-".repeat(40));
    console.log("📊 执行结果摘要：");
    console.log(`   总消息数：${result.messages.length}`);

    /**
     * 打印每个智能体的输出
     */
    for (const msg of result.messages) {
      if (isAIMessage(msg)) {
        const content = String(msg.content);
        if (content.startsWith("[Researcher]") || content.startsWith("[Writer]")) {
          const agentName = content.startsWith("[Researcher]") ? "研究员" : "写作者";
          console.log(`\n   --- ${agentName}输出 ---`);
          console.log(`   ${content.substring(0, 300)}${content.length > 300 ? "..." : ""}`);
        }
      }
    }
  } catch (error) {
    if (error instanceof GraphRecursionError) {
      console.error("❌ 达到最大递归次数，图执行被终止");
    } else {
      console.error("❌ 执行出错：", error);
    }
  }
}

/**
 * 演示 3：完整流水线 —— 研究 → 写作 → 审核
 *
 * 展示三个智能体如何按流水线协作
 * 场景：研究 AI 发展 → 撰写介绍文章 → 审核文章质量
 */
async function runFullPipelineDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 3：完整流水线 —— 研究 → 写作 → 审核");
  console.log("=".repeat(60));
  console.log("\n场景：研究人工智能发展历史，撰写介绍文章，并进行质量审核\n");

  const graph = createMultiAgentGraph();

  try {
    const result = await graph.invoke(
      {
        messages: [
          new HumanMessage(
            "请研究人工智能的发展历史，写一篇介绍文章，并审核文章质量。"
          ),
        ],
      },
      { recursionLimit: 20 }
    );

    console.log("\n" + "-".repeat(40));
    console.log("📊 执行结果摘要：");
    console.log(`   总消息数：${result.messages.length}`);

    /**
     * 打印每个智能体的输出摘要
     */
    for (const msg of result.messages) {
      if (isAIMessage(msg)) {
        const content = String(msg.content);
        const agentMatch = content.match(
          /^\[(Researcher|Writer|Reviewer|Supervisor)\]/
        );
        if (agentMatch) {
          const agentNames: Record<string, string> = {
            Researcher: "研究员",
            Writer: "写作者",
            Reviewer: "审核员",
            Supervisor: "调度员",
          };
          console.log(
            `\n   --- ${agentNames[agentMatch[1]] || agentMatch[1]}输出 ---`
          );
          console.log(
            `   ${content.substring(0, 300)}${content.length > 300 ? "..." : ""}`
          );
        }
      }
    }
  } catch (error) {
    if (error instanceof GraphRecursionError) {
      console.error("❌ 达到最大递归次数，图执行被终止");
    } else {
      console.error("❌ 执行出错：", error);
    }
  }
}

/**
 * 演示 4：流式执行 —— 实时查看智能体协作过程
 *
 * 展示如何使用 stream() 方法实时查看图的执行过程
 * 与 invoke() 一次性返回最终结果不同，stream() 会在每个节点执行后
 * 产出当前的状态快照，让我们可以实时观察智能体的协作过程
 */
async function runStreamingDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 4：流式执行 —— 实时查看协作过程");
  console.log("=".repeat(60));
  console.log("\n场景：流式查看搜索 DeepSeek 信息并撰写介绍的全过程\n");

  const graph = createMultiAgentGraph();

  try {
    /**
     * 使用 stream() 方法流式执行图
     * streamMode: "values" 表示每个步骤后输出完整的状态快照
     *
     * 其他 streamMode 选项：
     *   - "values"：输出完整状态
     *   - "updates"：只输出状态增量更新
     *   - "debug"：输出调试信息
     */
    const stream = await graph.stream(
      {
        messages: [
          new HumanMessage("搜索 DeepSeek 的信息，然后写一段简短的介绍。"),
        ],
      },
      {
        recursionLimit: 15,
        streamMode: "values",
      }
    );

    /**
     * 遍历流式输出
     * 每次迭代对应一个节点执行后的状态快照
     */
    let stepCount = 0;
    for await (const state of stream) {
      stepCount++;
      const lastMessage = state.messages[state.messages.length - 1];

      if (lastMessage && isAIMessage(lastMessage)) {
        const content = String(lastMessage.content);
        const agentMatch = content.match(
          /^\[(Researcher|Writer|Reviewer|Supervisor)\]/
        );

        if (agentMatch) {
          console.log(`\n   [步骤 ${stepCount}] ${agentMatch[1]} 已完成`);
        } else if (content.length > 0) {
          console.log(
            `\n   [步骤 ${stepCount}] AI 输出：${content.substring(0, 100)}${content.length > 100 ? "..." : ""}`
          );
        }
      }
    }

    console.log(`\n   总共执行了 ${stepCount} 个步骤`);
  } catch (error) {
    if (error instanceof GraphRecursionError) {
      console.error("❌ 达到最大递归次数，图执行被终止");
    } else {
      console.error("❌ 执行出错：", error);
    }
  }
}

/**
 * ============================================
 * 第九部分：主函数
 * ============================================
 */

/**
 * 主函数
 * 依次运行所有多智能体协作演示
 */
async function main() {
  console.log("LangChain Multi-Agent (多智能体协作) 功能演示");
  console.log("=".repeat(60));
  console.log("\n本演示展示如何使用 LangGraph 构建多智能体协作系统");
  console.log("多个智能体各司其职，通过 Supervisor 调度协同完成复杂任务\n");

  console.log("架构概览：");
  console.log("  Supervisor（调度员）→ Researcher（研究员）");
  console.log("                       → Writer（写作者）");
  console.log("                       → Reviewer（审核员）");
  console.log("");

  try {
    await runBasicRoutingDemo();

    console.log("\n等待 3 秒后继续下一个演示...\n");
    await new Promise((resolve) => setTimeout(resolve, 3000));

    await runResearchAndWriteDemo();

    console.log("\n等待 3 秒后继续下一个演示...\n");
    await new Promise((resolve) => setTimeout(resolve, 3000));

    await runFullPipelineDemo();

    console.log("\n等待 3 秒后继续下一个演示...\n");
    await new Promise((resolve) => setTimeout(resolve, 3000));

    await runStreamingDemo();

    console.log("\n" + "=".repeat(60));
    console.log("所有演示运行完成！");
    console.log("=".repeat(60));

    console.log("\n💡 关键要点回顾：");
    console.log("   1. LangGraph 的 StateGraph 用于构建有状态的多智能体工作流");
    console.log("   2. Annotation 定义状态结构，reducer 控制状态更新方式");
    console.log("   3. addConditionalEdges 实现动态路由，根据状态选择下一个节点");
    console.log("   4. Supervisor 模式通过中心调度实现智能体间的协调");
    console.log("   5. 每个智能体可以有独立的系统提示词和工具集");
    console.log("   6. stream() 方法支持实时查看执行过程");
  } catch (error) {
    console.error("运行出错：", error);
  }
}

main();
