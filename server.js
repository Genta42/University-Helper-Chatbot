// node --version # Should be >= 18
// npm install @google/generative-ai express

const express = require("express");
const { GoogleGenerativeAI, HarmCategory, HarmBlockThreshold } = require("@google/generative-ai");
const dotenv = require("dotenv").config();

const app = express();
const port = process.env.PORT || 3000;
app.use(express.json());
const MODEL_NAME = "gemini-pro";
const API_KEY = process.env.API_KEY;

// Store conversation history in memory
let conversationHistory = [
  {
    role: "user",
    parts: [
      {
        text: "Act as a University Resource Finder Chatbot for students. Your role is to: Answer questions and provide information about campus resources, services, events, and academic support available to students. Remember and retain details shared by the user throughout the conversation, such as their name, major, interests, and previously asked questions. Use this information to make responses more personalized. Examples of questions: 'Where can I find tutoring support for my courses?', 'How do I contact the financial aid office?', or 'What resources are available for mental health?' For unknown answers, suggest contacting the main student services desk.",
      },
    ],
  },
];

async function runChat(userInput) {
  const genAI = new GoogleGenerativeAI(API_KEY);
  const model = genAI.getGenerativeModel({ model: MODEL_NAME });

  const generationConfig = {
    temperature: 0.9,
    topK: 1,
    topP: 1,
    maxOutputTokens: 1000,
  };

  const safetySettings = [
    {
      category: HarmCategory.HARM_CATEGORY_HARASSMENT,
      threshold: HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
    },
  ];

  conversationHistory.push({
    role: "user",
    parts: [{ text: userInput }],
  });
// Conversation History
  const chat = model.startChat({
    generationConfig,
    safetySettings,
    history: conversationHistory, 
  });

  const result = await chat.sendMessage(userInput);
  const botResponse = result.response.text();

  conversationHistory.push({
    role: "model",
    parts: [{ text: botResponse }],
  });

  return botResponse;
}

app.get("/", (req, res) => {
  res.sendFile(__dirname + "/index.html");
});

app.get("/loader.gif", (req, res) => {
  res.sendFile(__dirname + "/loader.gif");
});

app.post("/chat", async (req, res) => {
  try {
    const userInput = req.body?.userInput;
    console.log("Incoming /chat request:", userInput);

    if (!userInput) {
      return res.status(400).json({ error: "Invalid request body" });
    }

    // Generate response
    const response = await runChat(userInput);

    res.json({ response });
  } catch (error) {
    console.error("Error in chat endpoint:", error);
    res.status(500).json({ error: "Internal Server Error" });
  }
});

app.listen(port, () => {
  console.log(`Server listening on port ${port}`);
});