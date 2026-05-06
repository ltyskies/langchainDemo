import { ChatOpenAI } from "@langchain/openai";
import dotenv from "dotenv";

dotenv.config();

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

const res = await model.invoke("用一句话解释什么是量子纠缠？");
console.log(res.content);
