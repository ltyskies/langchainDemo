import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document, DocumentInterface } from "@langchain/core/documents";
import { EmbeddingsInterface } from "@langchain/core/embeddings";
import dotenv from "dotenv";

// 加载环境变量配置
dotenv.config();

/**
 * 简单嵌入模型实现类
 * 实现了 EmbeddingsInterface 接口，用于将文本转换为向量表示
 * 使用基于字符的哈希算法生成简单的词嵌入向量
 */
class SimpleEmbeddings implements EmbeddingsInterface {
  /**
   * 将多个文档嵌入为向量
   * @param documents - 要嵌入的文档文本数组
   * @returns 返回每个文档对应的向量数组
   */
  async embedDocuments(documents: string[]): Promise<number[][]> {
    return documents.map((doc) => this.embedQuery(doc));
  }

  /**
   * 将单个查询文本嵌入为向量
   * @param document - 要嵌入的查询文本
   * @returns 返回100维的归一化向量
   * 
   * 实现原理：
   * 1. 将文本转换为小写并按空白字符分割成单词
   * 2. 初始化一个100维的零向量
   * 3. 遍历每个字符，根据字符的ASCII码对100取模来确定向量位置，并累加计数
   * 4. 对向量进行L2归一化处理
   */
  async embedQuery(document: string): Promise<number[]> {
    const words = document.toLowerCase().split(/\s+/);
    const vector: number[] = new Array(100).fill(0);

    for (const word of words) {
      for (let i = 0; i < word.length; i++) {
        const charCode = word.charCodeAt(i);
        vector[charCode % 100] += 1;
      }
    }

    const norm = Math.sqrt(vector.reduce((sum, val) => sum + val * val, 0));
    return norm > 0 ? vector.map((val) => val / norm) : vector;
  }
}

/**
 * 简单内存向量存储类
 * 用于在内存中存储文档向量并提供相似度搜索功能
 * 实现了基本的向量数据库功能，适用于小规模数据场景
 */
class SimpleMemoryVectorStore {
  // 存储所有文档的向量表示
  private vectors: number[][] = [];
  // 存储原始文档对象
  private documents: DocumentInterface[] = [];
  // 嵌入模型实例
  private embeddings: EmbeddingsInterface;

  /**
   * 构造函数
   * @param embeddings - 用于生成向量嵌入的模型实例
   */
  constructor(embeddings: EmbeddingsInterface) {
    this.embeddings = embeddings;
  }

  /**
   * 添加向量及其对应的文档到存储中
   * @param vectors - 要添加的向量数组
   * @param documents - 向量对应的文档数组
   */
  async addVectors(
    vectors: number[][],
    documents: DocumentInterface[]
  ): Promise<void> {
    this.vectors.push(...vectors);
    this.documents.push(...documents);
  }

  /**
   * 添加文档到存储中
   * 会自动将文档内容转换为向量
   * @param documents - 要添加的文档数组
   */
  async addDocuments(documents: DocumentInterface[]): Promise<void> {
    const texts = documents.map((doc) => doc.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);
    await this.addVectors(vectors, documents);
  }

  /**
   * 基于查询文本进行相似度搜索
   * @param query - 查询文本
   * @param k - 返回最相似的k个结果，默认为4
   * @returns 返回最相似的文档数组
   */
  async similaritySearch(query: string, k: number = 4): Promise<DocumentInterface[]> {
    const queryVector = await this.embeddings.embedQuery(query);
    const results = await this.similaritySearchVectorWithScore(queryVector, k);
    return results.map(([doc]) => doc);
  }

  /**
   * 基于向量进行相似度搜索并返回相似度分数
   * @param query - 查询向量
   * @param k - 返回最相似的k个结果
   * @returns 返回包含文档和相似度分数的元组数组
   */
  async similaritySearchVectorWithScore(
    query: number[],
    k: number
  ): Promise<[DocumentInterface, number][]> {
    // 计算查询向量与所有存储向量的余弦相似度
    const scores = this.vectors.map((vector) =>
      this.cosineSimilarity(query, vector)
    );

    // 将分数与索引关联并排序
    const indexedScores = scores.map((score, index) => ({ score, index }));
    indexedScores.sort((a, b) => b.score - a.score);

    // 取前k个结果
    const topK = indexedScores.slice(0, k);
    return topK.map(({ score, index }) => [this.documents[index], score]);
  }

  /**
   * 将向量存储转换为检索器对象
   * @param options - 配置选项，包含k值（返回结果数量）
   * @returns 返回一个具有invoke方法的对象，可用于检索
   */
  asRetriever(options: { k: number }): { invoke: (query: string) => Promise<DocumentInterface[]> } {
    const k = options.k;
    return {
      invoke: async (query: string) => {
        return this.similaritySearch(query, k);
      },
    };
  }

  /**
   * 清空向量存储中的所有数据
   */
  async delete(): Promise<void> {
    this.vectors = [];
    this.documents = [];
  }

  /**
   * 计算两个向量之间的余弦相似度
   * @param a - 第一个向量
   * @param b - 第二个向量
   * @returns 返回余弦相似度值，范围在-1到1之间
   * 
   * 余弦相似度 = (A·B) / (||A|| * ||B||)
   */
  private cosineSimilarity(a: number[], b: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

  /**
   * 静态工厂方法：从文档数组创建向量存储实例
   * @param docs - 要存储的文档数组
   * @param embeddings - 嵌入模型实例
   * @returns 返回已初始化的SimpleMemoryVectorStore实例
   */
  static async fromDocuments(
    docs: DocumentInterface[],
    embeddings: EmbeddingsInterface
  ): Promise<SimpleMemoryVectorStore> {
    const store = new SimpleMemoryVectorStore(embeddings);
    await store.addDocuments(docs);
    return store;
  }
}

// 初始化ChatOpenAI模型实例，配置使用DeepSeek API
const model = new ChatOpenAI({
  model: process.env.DEEPSEEK_MODEL,        // 模型名称
  apiKey: process.env.DEEPSEEK_API_KEY,     // API密钥
  configuration: {
    baseURL: process.env.DEEPSEEK_BASE_URL, // API基础URL
  },
  temperature: 0.7, // 温度参数，控制输出的随机性，0.7表示适度创造性
});

// 创建简单嵌入模型实例
const embeddings = new SimpleEmbeddings();

// 创建字符串输出解析器，用于将模型输出解析为字符串
const outputParser = new StringOutputParser();

/**
 * 知识库数据
 * 包含关于LangChain、RAG、向量数据库等AI技术的中文说明文档
 */
const knowledgeBase = [
  `LangChain 是一个用于开发由语言模型驱动的应用程序的框架。它提供了一系列工具和抽象，
  使开发者能够轻松地将语言模型与外部数据源、API 和其他工具集成。LangChain 的主要组件包括：
  模型 I/O（输入/输出管理）、检索（RAG）、链（Chains）、代理（Agents）和记忆（Memory）。`,

  `RAG（Retrieval-Augmented Generation，检索增强生成）是一种将信息检索与文本生成相结合的 AI 技术。
  它首先从知识库中检索相关信息，然后将这些信息作为上下文提供给语言模型，从而生成更准确、
  更相关的回答。RAG 可以有效解决语言模型幻觉问题和知识时效性问题。`,

  `向量数据库是一种专门用于存储和查询向量嵌入的数据库。它将文本、图像等数据转换为高维向量，
  并通过相似度搜索快速找到相关内容。常用的向量数据库包括 Pinecone、Weaviate、Chroma 等。
  LangChain 提供了与这些向量数据库的集成支持。`,

  `LangChain 的 Chain（链）概念允许将多个组件组合成一个可执行的工作流。例如，可以将文档加载、
  文本分割、嵌入生成、向量存储和检索等步骤链接在一起，构建一个完整的 RAG 管道。
  这种模块化设计使得应用开发和维护更加容易。`,

  `嵌入（Embeddings）是将文本转换为数值向量的技术，使得语义相似的文本在向量空间中距离更近。
  OpenAI 提供了强大的嵌入模型，如 text-embedding-ada-002。LangChain 支持多种嵌入模型，
  并提供了统一的接口来使用它们。`,
];

/**
 * 简单文本分割器
 * 将长文本分割成指定大小的文档块
 * @param texts - 要分割的文本数组
 * @param chunkSize - 每个文档块的最大字符数，默认为500
 * @returns 返回分割后的Document对象数组
 */
function simpleTextSplitter(texts: string[], chunkSize: number = 500): Document[] {
  const docs: Document[] = [];
  for (const text of texts) {
    // 清理文本：将多个空白字符替换为单个空格并去除首尾空白
    const cleanedText = text.replace(/\s+/g, ' ').trim();
    if (cleanedText.length <= chunkSize) {
      // 如果文本长度小于等于块大小，直接作为一个文档
      docs.push(new Document({ pageContent: cleanedText }));
    } else {
      // 否则按块大小分割文本
      for (let i = 0; i < cleanedText.length; i += chunkSize) {
        const chunk = cleanedText.slice(i, i + chunkSize);
        docs.push(new Document({ pageContent: chunk }));
      }
    }
  }
  return docs;
}

/**
 * 运行基础RAG演示
 * 演示基本的检索增强生成流程：
 * 1. 文档分割
 * 2. 向量存储创建
 * 3. 相似度检索
 * 4. 使用检索结果作为上下文生成回答
 */
async function runBasicRAGDemo() {
  console.log("=== 基础 RAG 演示 ===\n");

  // 将知识库文本分割成文档块
  const docs = simpleTextSplitter(knowledgeBase);
  console.log(`文档分割完成，共 ${docs.length} 个文档块\n`);

  // 创建向量存储，将文档转换为向量并存储
  const vectorStore = await SimpleMemoryVectorStore.fromDocuments(docs, embeddings);
  console.log("向量存储创建完成\n");

  // 创建检索器，设置返回最相似的2个文档
  const retriever = vectorStore.asRetriever({ k: 2 });

  // 定义RAG提示词模板
  const ragPrompt = ChatPromptTemplate.fromMessages([
    [
      "system",
      `你是一个专业的 AI 助手。请根据以下检索到的上下文信息回答用户的问题。
如果上下文中没有相关信息，请坦诚告知。`,
    ],
    ["human", `上下文信息：
{context}

用户问题：{question}`],
  ]);

  const question = "什么是 RAG 技术？";
  console.log(`用户问题：${question}\n`);

  // 检索与问题相关的文档
  const relevantDocs = await retriever.invoke(question);
  console.log("检索到的相关文档：");
  relevantDocs.forEach((doc, i) => {
    console.log(`\n[文档 ${i + 1}]`);
    console.log(doc.pageContent.substring(0, 150) + "...");
  });

  // 将检索到的文档内容合并为上下文
  const context = relevantDocs.map((doc) => doc.pageContent).join("\n\n");

  // 构建处理链：提示词 -> 模型 -> 输出解析器
  const chain = ragPrompt.pipe(model).pipe(outputParser);
  const response = await chain.invoke({
    context,
    question,
  });

  console.log("\n\nAI 回答：");
  console.log(response);
}

/**
 * 运行多查询RAG演示
 * 演示对多个问题依次进行RAG检索和回答
 */
async function runMultiQueryRAGDemo() {
  console.log("\n\n=== 多查询 RAG 演示 ===\n");

  const docs = simpleTextSplitter(knowledgeBase);
  const vectorStore = await SimpleMemoryVectorStore.fromDocuments(docs, embeddings);
  const retriever = vectorStore.asRetriever({ k: 2 });

  // 定义多个测试问题
  const questions = [
    "LangChain 的主要组件有哪些？",
    "向量数据库有什么作用？",
    "嵌入技术是什么？",
  ];

  // 对每个问题执行RAG流程
  for (const question of questions) {
    console.log(`\n--- 问题：${question} ---\n`);

    const relevantDocs = await retriever.invoke(question);
    const context = relevantDocs.map((doc) => doc.pageContent).join("\n\n");

    const ragPrompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "你是一个专业的 AI 助手。请根据上下文简洁准确地回答用户问题。",
      ],
      ["human", `上下文：\n{context}\n\n问题：{question}`],
    ]);

    const chain = ragPrompt.pipe(model).pipe(outputParser);
    const response = await chain.invoke({
      context,
      question,
    });

    console.log(`回答：${response}\n`);
  }
}

/**
 * 运行自定义文档RAG演示
 * 演示使用带有元数据的自定义文档进行RAG
 * 展示如何处理结构化数据（如产品信息）
 */
async function runCustomDocumentRAGDemo() {
  console.log("\n\n=== 自定义文档 RAG 演示 ===\n");

  // 创建带有元数据的自定义文档
  const customDocs = [
    new Document({
      pageContent: "产品名称：智能助手 Pro\n发布日期：2024年1月\n价格：￥999/年",
      metadata: { source: "产品手册", category: "产品信息" },
    }),
    new Document({
      pageContent:
        "智能助手 Pro 支持多语言对话、文档分析、代码生成等功能。适用于企业办公、教育学习、软件开发等场景。",
      metadata: { source: "功能介绍", category: "功能特性" },
    }),
    new Document({
      pageContent:
        "退款政策：购买后7天内可申请全额退款。超过7天但不满30天，可申请部分退款。",
      metadata: { source: "服务条款", category: "售后政策" },
    }),
  ];

  const vectorStore = await SimpleMemoryVectorStore.fromDocuments(customDocs, embeddings);
  const retriever = vectorStore.asRetriever({ k: 1 });

  // 定义针对自定义文档的测试问题
  const questions = [
    "智能助手 Pro 多少钱？",
    "这个产品有什么功能？",
    "退款政策是什么？",
  ];

  for (const question of questions) {
    console.log(`问题：${question}`);

    const relevantDocs = await retriever.invoke(question);

    // 显示检索结果的来源信息
    console.log("检索来源：");
    relevantDocs.forEach((doc) => {
      console.log(`  - ${doc.metadata.source} (${doc.metadata.category})`);
    });

    const context = relevantDocs.map((doc) => doc.pageContent).join("\n");

    const ragPrompt = ChatPromptTemplate.fromMessages([
      [
        "system",
        "你是智能助手 Pro 的客服机器人。请根据提供的信息回答用户问题。",
      ],
      ["human", `相关信息：\n{context}\n\n用户问题：{question}`],
    ]);

    const chain = ragPrompt.pipe(model).pipe(outputParser);
    const response = await chain.invoke({
      context,
      question,
    });

    console.log(`回答：${response}\n`);
  }
}

/**
 * 主函数
 * 依次运行三个RAG演示函数
 */
async function main() {
  try {
    await runBasicRAGDemo();
    await runMultiQueryRAGDemo();
    await runCustomDocumentRAGDemo();
  } catch (error) {
    console.error("运行出错：", error);
  }
}

// 执行主函数
main();
