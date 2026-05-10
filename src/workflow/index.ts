/**
 * ============================================
 * LangChain Workflow (工作流) 功能演示
 * ============================================
 *
 * 本演示展示了如何使用 LangChain + LangGraph 构建各种工作流模式。
 * 工作流是将多个处理步骤按特定逻辑组织起来，形成一个可执行的管道。
 *
 * 本 demo 包含以下内容：
 *   1. 工作流核心概念介绍
 *   2. 顺序管道 (Sequential Pipeline)
 *   3. 条件分支 (Conditional Branching)
 *   4. 并行扇出/扇入 (Fan-Out / Fan-In)
 *   5. 循环迭代 (Loop / Cyclic)
 *   6. 流式执行与可视化
 *
 * LangGraph 关键概念：
 *   - StateGraph：有向图，节点之间通过共享状态通信
 *   - Annotation：定义状态的结构和更新方式
 *   - Node：图中的处理单元
 *   - Edge：节点之间的固定连接
 *   - Conditional Edge：根据状态动态选择下一个节点
 *   - START / END：图的入口和出口
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
} from "@langchain/langgraph";
import {
  HumanMessage,
  AIMessage,
  SystemMessage,
  BaseMessage,
} from "@langchain/core/messages";
import { tool } from "@langchain/core/tools";
import { z } from "zod";
import dotenv from "dotenv";

dotenv.config();

/**
 * ============================================
 * 第一部分：模型与工具初始化
 * ============================================
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
 * 模拟搜索工具
 */
const searchTool = tool(
  async ({ query }: { query: string }) => {
    const results: Record<string, string> = {
      "人工智能": "2025年AI领域发展迅速，大语言模型、多模态AI和AI Agent成为三大热点方向。",
      "量子计算": "量子计算利用量子比特进行计算，2025年在纠错和实用性方面取得重大突破。",
      "新能源": "2025年全球可再生能源占比持续提升，光伏和储能技术成本大幅下降。",
      "云计算": "云计算市场持续增长，边缘计算和Serverless架构成为新趋势。",
    };

    for (const [key, value] of Object.entries(results)) {
      if (query.toLowerCase().includes(key.toLowerCase())) {
        return `搜索结果：${value}`;
      }
    }
    return `搜索结果：关于"${query}"的相关信息摘要。`;
  },
  {
    name: "search",
    description: "搜索互联网获取信息",
    schema: z.object({
      query: z.string().describe("搜索关键词"),
    }),
  }
);

/**
 * ============================================
 * 第二部分：顺序管道工作流
 * ============================================
 *
 * 场景：文章生成管道
 *   输入主题 → 研究收集 → 大纲规划 → 内容撰写 → 润色输出
 *
 * 这是最简单的工作流模式，各步骤按固定顺序依次执行。
 */

const PipelineState = Annotation.Root({
  topic: Annotation<string>,
  researchNotes: Annotation<string>,
  outline: Annotation<string>,
  draft: Annotation<string>,
  finalArticle: Annotation<string>,
});

async function researchNode(state: typeof PipelineState.State) {
  console.log("  [1/4] 研究收集：搜索相关资料...");

  const response = await model.invoke([
    new SystemMessage("你是一个专业的研究员。请为以下主题收集3-5个关键信息点，用简洁的要点列出。"),
    new HumanMessage(`主题：${state.topic}`),
  ]);

  const researchNotes = String(response.content);
  console.log(`  ✓ 收集到 ${researchNotes.length} 字符的研究资料`);

  return { researchNotes };
}

async function outlineNode(state: typeof PipelineState.State) {
  console.log("  [2/4] 大纲规划：设计文章结构...");

  const response = await model.invoke([
    new SystemMessage("你是一个专业的文章策划师。请根据研究资料设计文章大纲，包含标题和各章节要点。"),
    new HumanMessage(`研究资料：\n${state.researchNotes}\n\n请为该主题设计文章大纲。`),
  ]);

  const outline = String(response.content);
  console.log(`  ✓ 大纲已生成`);

  return { outline };
}

async function writeNode(state: typeof PipelineState.State) {
  console.log("  [3/4] 内容撰写：根据大纲撰写文章...");

  const response = await model.invoke([
    new SystemMessage("你是一个专业的内容创作者。请根据大纲撰写一篇完整、流畅的文章。"),
    new HumanMessage(
      `主题：${state.topic}\n研究资料：\n${state.researchNotes}\n大纲：\n${state.outline}\n\n请撰写完整的文章内容。`
    ),
  ]);

  const draft = String(response.content);
  console.log(`  ✓ 初稿完成，共 ${draft.length} 字符`);

  return { draft };
}

async function polishNode(state: typeof PipelineState.State) {
  console.log("  [4/4] 润色优化：打磨文章...");

  const response = await model.invoke([
    new SystemMessage("你是一个专业的编辑。请在保持原意的基础上，润色以下文章，使其更加流畅、生动、易读。"),
    new HumanMessage(`原文：\n${state.draft}\n\n请输出润色后的最终版本。`),
  ]);

  const finalArticle = String(response.content);
  console.log(`  ✓ 最终文章已生成，共 ${finalArticle.length} 字符`);

  return { finalArticle };
}

function createPipelineGraph() {
  return new StateGraph(PipelineState)
    .addNode("research", researchNode)
    .addNode("outline", outlineNode)
    .addNode("write", writeNode)
    .addNode("polish", polishNode)
    .addEdge(START, "research")
    .addEdge("research", "outline")
    .addEdge("outline", "write")
    .addEdge("write", "polish")
    .addEdge("polish", END)
    .compile();
}

/**
 * ============================================
 * 第三部分：条件分支工作流
 * ============================================
 *
 * 场景：智能客服工单路由
 *   用户提交工单 → 分析意图 → 按类型分发 → 专业技术/账单/通用 → 生成回复
 *
 * 条件分支根据状态动态决定下一步，实现灵活的路由逻辑。
 */

const TicketState = Annotation.Root({
  messages: Annotation<BaseMessage[]>({
    reducer: messagesStateReducer,
    default: () => [],
  }),
  category: Annotation<string>,
  response: Annotation<string>,
});

async function analyzeNode(state: typeof TicketState.State) {
  console.log("  [分析] 分析用户意图...");

  const response = await model.invoke([
    new SystemMessage(
      `你是一个客服工单分析员。请分析用户的问题，判断类别并输出。
只输出以下类别之一：technical（技术问题）、billing（账单问题）、general（通用问题）`
    ),
    ...state.messages,
  ]);

  const category = String(response.content).trim().toLowerCase();
  const normalizedCategory = category.includes("technical")
    ? "technical"
    : category.includes("billing")
      ? "billing"
      : "general";

  console.log(`  ✓ 分类结果：${normalizedCategory}`);

  return {
    category: normalizedCategory,
    messages: [new AIMessage(`[分类] ${normalizedCategory}`)],
  };
}

async function technicalNode(state: typeof TicketState.State) {
  console.log("  [技术客服] 处理技术问题...");

  const techModel = model.bindTools([searchTool]);
  const response = await techModel.invoke([
    new SystemMessage("你是技术客服专家，请先搜索相关技术知识，然后给出专业、详细的解答。"),
    ...state.messages,
  ]);

  return {
    response: `[技术客服]\n${response.content}`,
    messages: [new AIMessage(`[技术客服]\n${response.content}`)],
  };
}

async function billingNode(state: typeof TicketState.State) {
  console.log("  [账单客服] 处理账单问题...");

  const response = await model.invoke([
    new SystemMessage("你是账单客服专家，请专业、耐心地解答账单相关问题。如果涉及退款，请说明退款政策和流程。"),
    ...state.messages,
  ]);

  return {
    response: `[账单客服]\n${response.content}`,
    messages: [new AIMessage(`[账单客服]\n${response.content}`)],
  };
}

async function generalNode(state: typeof TicketState.State) {
  console.log("  [通用客服] 处理一般问题...");

  const response = await model.invoke([
    new SystemMessage("你是通用客服，请友好、简洁地回答用户的一般性问题。"),
    ...state.messages,
  ]);

  return {
    response: `[通用客服]\n${response.content}`,
    messages: [new AIMessage(`[通用客服]\n${response.content}`)],
  };
}

function categoryRouter(state: typeof TicketState.State): string {
  return state.category;
}

function createTicketGraph() {
  return new StateGraph(TicketState)
    .addNode("analyze", analyzeNode)
    .addNode("technical", technicalNode)
    .addNode("billing", billingNode)
    .addNode("general", generalNode)
    .addEdge(START, "analyze")
    .addConditionalEdges("analyze", categoryRouter, {
      technical: "technical",
      billing: "billing",
      general: "general",
    })
    .addEdge("technical", END)
    .addEdge("billing", END)
    .addEdge("general", END)
    .compile();
}

/**
 * ============================================
 * 第四部分：并行扇出/扇入工作流
 * ============================================
 *
 * 场景：多源研究汇总
 *   用户提问 → 并行搜索多个信息来源 → 汇总整合 → 输出综合报告
 *
 * Send() API 允许向多个节点并行发送任务，是 LangGraph 的扇出机制。
 * 这里演示使用条件边模拟多源并行查找。
 */

const ResearchState = Annotation.Root({
  question: Annotation<string>,
  sources: Annotation<string[]>,
  results: Annotation<string[]>,
  finalReport: Annotation<string>,
});

async function dispatchNode(state: typeof ResearchState.State) {
  console.log("  [调度] 确定需要查询的信息源...");

  const response = await model.invoke([
    new SystemMessage(
      "你是一个研究调度员。根据用户问题，输出3个不同的搜索角度（用换行分隔），每个角度一行，不要编号。"
    ),
    new HumanMessage(`问题：${state.question}`),
  ]);

  const sources = String(response.content)
    .split("\n")
    .map((s) => s.replace(/^\d+[\.\、\)]\s*/, "").trim())
    .filter((s) => s.length > 0)
    .slice(0, 3);

  console.log(`  ✓ 确定 ${sources.length} 个搜索角度：`);
  sources.forEach((s, i) => console.log(`    ${i + 1}. ${s}`));

  return { sources };
}

async function parallelSearchNode(state: typeof ResearchState.State) {
  // 并发执行所有搜索
  const searchPromises = state.sources.map((query, index) =>
    searchTool.invoke({ query } as any).then((result) => {
      console.log(`  [搜索 ${index + 1}] "${query}" → 完成`);
      return `[来源${index + 1}: ${query}]\n${result}`;
    })
  );

  const results = await Promise.all(searchPromises);
  console.log(`  ✓ 所有搜索并行完成`);

  return { results };
}

async function synthesizeNode(state: typeof ResearchState.State) {
  console.log("  [整合] 汇总多方信息...");

  const combinedResults = state.results.map((r, i) => `--- 来源 ${i + 1} ---\n${r}`).join("\n\n");

  const response = await model.invoke([
    new SystemMessage("你是一个高级研究分析师。请综合以下多个来源的信息，撰写一个全面、有条理的综合报告。"),
    new HumanMessage(`用户问题：${state.question}\n\n各来源信息：\n${combinedResults}\n\n请输出综合报告。`),
  ]);

  const report = String(response.content);
  console.log(`  ✓ 综合报告已生成，共 ${report.length} 字符`);

  return { finalReport: report };
}

function createResearchGraph() {
  return new StateGraph(ResearchState)
    .addNode("dispatch", dispatchNode)
    .addNode("parallelSearch", parallelSearchNode)
    .addNode("synthesize", synthesizeNode)
    .addEdge(START, "dispatch")
    .addEdge("dispatch", "parallelSearch")
    .addEdge("parallelSearch", "synthesize")
    .addEdge("synthesize", END)
    .compile();
}

/**
 * ============================================
 * 第五部分：循环迭代工作流
 * ============================================
 *
 * 场景：质量审核循环
 *   生成初稿 → 质量审核 → 得分 < 80？→ 改进 → 再次审核 → 得分 >= 80 → 输出
 *
 * 循环是工作流中的重要模式，确保输出达到质量标准。
 */

const LoopState = Annotation.Root({
  task: Annotation<string>,
  content: Annotation<string>,
  score: Annotation<number>,
  iteration: Annotation<number>,
});

async function generateNode(state: typeof LoopState.State) {
  const iteration = (state.iteration || 0) + 1;
  console.log(`  [第${iteration}轮] 生成/改进内容...`);

  const isFirstRound = !state.content;

  const messages: BaseMessage[] = [
    new SystemMessage(
      isFirstRound
        ? "你是一个专业的内容创作者。请根据任务要求生成一段内容。"
        : `你是一个专业的内容创作者。请根据审核反馈改进以下内容。当前内容得分：${state.score}/100，目标是80分以上。请认真改进。`
    ),
  ];

  if (isFirstRound) {
    messages.push(new HumanMessage(`任务：${state.task}\n请生成一段高质量的内容。`));
  } else {
    messages.push(new HumanMessage(`任务：${state.task}\n当前内容：\n${state.content}\n\n请根据反馈改进内容，确保质量。`));
  }

  const response = await model.invoke(messages);
  const content = String(response.content);
  console.log(`  ✓ 内容已生成/改进，共 ${content.length} 字符`);

  return { content, iteration };
}

async function reviewNode(state: typeof LoopState.State) {
  console.log(`  [审核] 评估内容质量...`);

  const response = await model.invoke([
    new SystemMessage(
      `你是一个严格的内容审核员。请根据以下标准评分（0-100分）：
- 内容完整性（25分）
- 逻辑清晰度（25分）
- 表达流畅度（25分）
- 与任务的匹配度（25分）

请严格评分，并在输出末尾以 "分数: XX" 的格式给出最终得分。`
    ),
    new HumanMessage(`任务：${state.task}\n内容：\n${state.content}\n\n请审核并评分。`),
  ]);

  const reviewText = String(response.content);
  const scoreMatch = reviewText.match(/分数[:\s]*(\d+)/i);
  const score = scoreMatch ? parseInt(scoreMatch[1], 10) : 75;

  console.log(`  ✓ 审核完成，得分：${score}/100 ${score >= 80 ? "✅ 通过" : "❌ 需改进"}`);

  return { score };
}

function qualityRouter(state: typeof LoopState.State): string {
  if (state.score >= 80) {
    return "END";
  }
  if (state.iteration >= 3) {
    // 最多3轮，强制结束
    return "END";
  }
  return "generate";
}

function createQualityLoopGraph() {
  return new StateGraph(LoopState)
    .addNode("generate", generateNode)
    .addNode("review", reviewNode)
    .addEdge(START, "generate")
    .addEdge("generate", "review")
    .addConditionalEdges("review", qualityRouter, {
      generate: "generate",
      END: END,
    })
    .compile();
}

/**
 * ============================================
 * 第六部分：演示函数
 * ============================================
 */

/**
 * 演示 1：顺序管道
 */
async function runPipelineDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 1：顺序管道 —— 文章生成流水线");
  console.log("=".repeat(60));
  console.log("\n流程：研究 → 大纲 → 撰写 → 润色\n");

  const graph = createPipelineGraph();

  const result = await graph.invoke({
    topic: "2025年人工智能发展趋势",
  });

  console.log("\n" + "-".repeat(40));
  console.log("📄 最终文章：");
  console.log(result.finalArticle);
  console.log("-".repeat(40));
}

/**
 * 演示 2：条件分支
 */
async function runBranchingDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 2：条件分支 —— 智能客服工单路由");
  console.log("=".repeat(60));

  const testCases = [
    "我的服务器无法连接，总是显示502错误，请问怎么解决？",
    "我想问一下上个月的账单为什么多扣了50元？",
    "你们公司的办公时间是几点到几点？",
  ];

  const graph = createTicketGraph();

  for (const [index, query] of testCases.entries()) {
    console.log(`\n  --- 工单 ${index + 1} ---`);
    console.log(`  用户：${query}`);

    const result = await graph.invoke({
      messages: [new HumanMessage(query)],
    });

    console.log(`  类别：${result.category}`);
    console.log(`  回复：${(result.response || "").substring(0, 150)}...`);
  }

  console.log("\n  ✓ 三个工单已分别路由到对应的客服处理");
}

/**
 * 演示 3：并行扇入扇出
 */
async function runParallelDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 3：并行扇出/扇入 —— 多源研究汇总");
  console.log("=".repeat(60));
  console.log("\n流程：调度搜索角度 → 并行搜索多个来源 → 汇总整合\n");

  const graph = createResearchGraph();

  const result = await graph.invoke({
    question: "2025年科技行业有哪些重要发展趋势？",
  });

  console.log("\n" + "-".repeat(40));
  console.log("📊 综合研究报告：");
  console.log(result.finalReport);
  console.log("-".repeat(40));
}

/**
 * 演示 4：循环迭代
 */
async function runLoopDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 4：循环迭代 —— 质量审核循环");
  console.log("=".repeat(60));
  console.log("\n流程：生成 → 审核 → (得分 < 80?) → 改进 → 审核 → ... → 达标输出\n");

  const graph = createQualityLoopGraph();

  const result = await graph.invoke({
    task: "写一段关于人工智能伦理重要性的短文（约200字）",
    iteration: 0,
  });

  console.log("\n" + "-".repeat(40));
  console.log(`📝 最终内容（第${result.iteration}轮，得分 ${result.score}/100）：`);
  console.log(result.content);
  console.log("-".repeat(40));
}

/**
 * 演示 5：流式执行可视化
 */
async function runStreamingDemo() {
  console.log("\n" + "=".repeat(60));
  console.log("演示 5：流式执行 —— 实时可视化工作流执行过程");
  console.log("=".repeat(60));
  console.log("\n使用 stream() 方法实时观察每个节点的执行状态\n");

  const graph = createPipelineGraph();

  const stream = await graph.stream(
    { topic: "云计算与边缘计算" },
    { streamMode: "updates" }
  );

  console.log("实时执行状态：");
  let stepCount = 0;

  for await (const chunk of stream) {
    stepCount++;
    const nodeNames = Object.keys(chunk);

    for (const nodeName of nodeNames) {
      const updates = (chunk as Record<string, Record<string, unknown>>)[nodeName];
      const keyFields = Object.keys(updates);
      console.log(`  [步骤 ${stepCount}] 节点 "${nodeName}" 完成，更新字段：${keyFields.join(", ")}`);
    }
  }

  console.log(`\n  ✓ 工作流共经过 ${stepCount} 个步骤完成`);
}

/**
 * ============================================
 * 第七部分：主函数
 * ============================================
 */

async function main() {
  console.log("LangChain Workflow (工作流) 功能演示");
  console.log("=".repeat(60));
  console.log("\n本演示展示 LangGraph 的四种核心工作流模式：");
  console.log("  1. 顺序管道 ── 固定步骤依次执行");
  console.log("  2. 条件分支 ── 根据状态动态路由");
  console.log("  3. 并行扇出/扇入 ── 多节点并行处理");
  console.log("  4. 循环迭代 ── 条件不满足时重复执行");
  console.log("");

  try {
    // 演示 1：顺序管道
    await runPipelineDemo();
    await delay(3000);

    // 演示 2：条件分支
    await runBranchingDemo();
    await delay(3000);

    // 演示 3：并行扇入扇出
    await runParallelDemo();
    await delay(3000);

    // 演示 4：循环迭代
    await runLoopDemo();
    await delay(3000);

    // 演示 5：流式执行
    await runStreamingDemo();

    console.log("\n" + "=".repeat(60));
    console.log("所有演示运行完成！");
    console.log("=".repeat(60));

    console.log("\n💡 关键要点回顾：");
    console.log("  1. StateGraph + Annotation 定义工作流的状态结构");
    console.log("  2. addEdge 定义固定流转，addConditionalEdges 实现动态路由");
    console.log("  3. 并行处理通过 Promise.all 在节点内实现");
    console.log("  4. 循环通过条件边返回上游节点实现");
    console.log("  5. stream() 方法支持实时观察执行过程");
  } catch (error) {
    console.error("运行出错：", error);
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

main();
