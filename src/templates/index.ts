import { ChatOpenAI } from "@langchain/openai";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { StringOutputParser } from "@langchain/core/output_parsers";
import dotenv from "dotenv";

dotenv.config();

const model = new ChatOpenAI({
  model: process.env.DEEPSEEK_MODEL,
  apiKey: process.env.DEEPSEEK_API_KEY,
  configuration: {
    baseURL: process.env.DEEPSEEK_BASE_URL,
  },
  temperature: 0.7,
});

const outputParser = new StringOutputParser();

async function runTranslationDemo() {
  const translationTemplate = ChatPromptTemplate.fromMessages([
    [
      "system",
      "你是一位专业的翻译官。请将用户输入的文本翻译成{target_language}。",
    ],
    ["human", "{text}"],
  ]);

  const chain = translationTemplate.pipe(model).pipe(outputParser);

  const result = await chain.invoke({
    target_language: "英文",
    text: "你好，世界！这是一个使用 LangChain 提示词模板的示例。",
  });

  console.log("翻译结果：");
  console.log(result);
}

async function runCreativeWritingDemo() {
  const creativeTemplate = ChatPromptTemplate.fromMessages([
    [
      "system",
      "你是一位创意写作助手。请根据给定的主题和风格创作一段内容。",
    ],
    [
      "human",
      "主题：{topic}\n风格：{style}\n字数要求：{word_count}字左右",
    ],
  ]);

  const chain = creativeTemplate.pipe(model).pipe(outputParser);

  const result = await chain.invoke({
    topic: "未来城市",
    style: "科幻",
    word_count: "200",
  });

  console.log("\n创意写作结果：");
  console.log(result);
}

async function runCodeReviewDemo() {
  const codeReviewTemplate = ChatPromptTemplate.fromMessages([
    [
      "system",
      "你是一位资深程序员。请对以下代码进行审查，指出潜在问题和改进建议。",
    ],
    ["human", "编程语言：{language}\n代码：\n```{language}\n{code}\n```"],
  ]);

  const chain = codeReviewTemplate.pipe(model).pipe(outputParser);

  const sampleCode = `
function add(a, b) {
  return a + b;
}

console.log(add(1, 2));
`;

  const result = await chain.invoke({
    language: "javascript",
    code: sampleCode,
  });

  console.log("\n代码审查结果：");
  console.log(result);
}

async function main() {
  try {
    await runTranslationDemo();
    await runCreativeWritingDemo();
    await runCodeReviewDemo();
  } catch (error) {
    console.error("运行出错：", error);
  }
}

main();
