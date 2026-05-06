import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import { Document, DocumentInterface } from "@langchain/core/documents";
import { EmbeddingsInterface } from "@langchain/core/embeddings";
import dotenv from "dotenv";

dotenv.config();

class SimpleEmbeddings implements EmbeddingsInterface {
  async embedDocuments(documents: string[]): Promise<number[][]> {
    return documents.map((doc) => this.embedQuery(doc));
  }


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

class SimpleMemoryVectorStore {
  private vectors: number[][] = [];
  private documents: DocumentInterface[] = [];
  private embeddings: EmbeddingsInterface;

  constructor(embeddings: EmbeddingsInterface) {
    this.embeddings = embeddings;
  }

  async addVectors(
    vectors: number[][],
    documents: DocumentInterface[]
  ): Promise<void> {
    this.vectors.push(...vectors);
    this.documents.push(...documents);
  }

  async addDocuments(documents: DocumentInterface[]): Promise<void> {
    const texts = documents.map((doc) => doc.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);
    await this.addVectors(vectors, documents);
  }

  async similaritySearch(query: string, k: number = 4): Promise<DocumentInterface[]> {
    const queryVector = await this.embeddings.embedQuery(query);
    const results = await this.similaritySearchVectorWithScore(queryVector, k);
    return results.map(([doc]) => doc);
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number
  ): Promise<[DocumentInterface, number][]> {
    const scores = this.vectors.map((vector) =>
      this.cosineSimilarity(query, vector)
    );

    const indexedScores = scores.map((score, index) => ({ score, index }));
    indexedScores.sort((a, b) => b.score - a.score);

    const topK = indexedScores.slice(0, k);
    return topK.map(({ score, index }) => [this.documents[index], score]);
  }

  asRetriever(options: { k: number }): { invoke: (query: string) => Promise<DocumentInterface[]> } {
    const k = options.k;
    return {
      invoke: async (query: string) => {
        return this.similaritySearch(query, k);
      },
    };
  }

  async delete(): Promise<void> {
    this.vectors = [];
    this.documents = [];
  }

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

  static async fromDocuments(
    docs: DocumentInterface[],
    embeddings: EmbeddingsInterface
  ): Promise<SimpleMemoryVectorStore> {
    const store = new SimpleMemoryVectorStore(embeddings);
    await store.addDocuments(docs);
    return store;
  }
}

const model = new ChatOpenAI({
  model: process.env.DEEPSEEK_MODEL,
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: {
    baseURL: process.env.DEEPSEEK_BASE_URL,
  },
  temperature: 0.7,
});

const embeddings = new SimpleEmbeddings();

const outputParser = new StringOutputParser();

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

function simpleTextSplitter(texts: string[], chunkSize: number = 500): Document[] {
  const docs: Document[] = [];
  for (const text of texts) {
    const cleanedText = text.replace(/\s+/g, ' ').trim();
    if (cleanedText.length <= chunkSize) {
      docs.push(new Document({ pageContent: cleanedText }));
    } else {
      for (let i = 0; i < cleanedText.length; i += chunkSize) {
        const chunk = cleanedText.slice(i, i + chunkSize);
        docs.push(new Document({ pageContent: chunk }));
      }
    }
  }
  return docs;
}

async function runBasicRAGDemo() {
  console.log("=== 基础 RAG 演示 ===\n");

  const docs = simpleTextSplitter(knowledgeBase);
  console.log(`文档分割完成，共 ${docs.length} 个文档块\n`);

  const vectorStore = await SimpleMemoryVectorStore.fromDocuments(docs, embeddings);
  console.log("向量存储创建完成\n");

  const retriever = vectorStore.asRetriever({ k: 2 });

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

  const relevantDocs = await retriever.invoke(question);
  console.log("检索到的相关文档：");
  relevantDocs.forEach((doc, i) => {
    console.log(`\n[文档 ${i + 1}]`);
    console.log(doc.pageContent.substring(0, 150) + "...");
  });

  const context = relevantDocs.map((doc) => doc.pageContent).join("\n\n");

  const chain = ragPrompt.pipe(model).pipe(outputParser);
  const response = await chain.invoke({
    context,
    question,
  });

  console.log("\n\nAI 回答：");
  console.log(response);
}

async function runMultiQueryRAGDemo() {
  console.log("\n\n=== 多查询 RAG 演示 ===\n");

  const docs = simpleTextSplitter(knowledgeBase);
  const vectorStore = await SimpleMemoryVectorStore.fromDocuments(docs, embeddings);
  const retriever = vectorStore.asRetriever({ k: 2 });

  const questions = [
    "LangChain 的主要组件有哪些？",
    "向量数据库有什么作用？",
    "嵌入技术是什么？",
  ];

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

async function runCustomDocumentRAGDemo() {
  console.log("\n\n=== 自定义文档 RAG 演示 ===\n");

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

  const questions = [
    "智能助手 Pro 多少钱？",
    "这个产品有什么功能？",
    "退款政策是什么？",
  ];

  for (const question of questions) {
    console.log(`问题：${question}`);

    const relevantDocs = await retriever.invoke(question);

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

async function main() {
  try {
    await runBasicRAGDemo();
    await runMultiQueryRAGDemo();
    await runCustomDocumentRAGDemo();
  } catch (error) {
    console.error("运行出错：", error);
  }
}

main();
