import 'dotenv/config'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serve } from '@hono/node-server'
import { GoogleGenerativeAI } from '@google/generative-ai'
import OpenAI from 'openai'

const app = new Hono()
 app.use('/*', cors())

// เตรียม Client ของแต่ละค่าย
const geminiAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '')
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || '',
  baseURL: 'https://openrouter.ai/api/v1',
})

app.get('/', (c) => {
  return c.text('AI Router Server is Ready!')
})

app.post('/api/chat', async (c) => {
  try {
    const body = await c.req.json()
    const userPrompt = body.prompt

    if (!userPrompt) {
      return c.json({ error: 'Please provide a prompt' }, 400)
    }

    const lowerMsg = userPrompt.toLowerCase()
    let category = 'GENERAL'
    let modelUsed = ''
    let replyText = ''

    if (lowerMsg.includes('code') || lowerMsg.includes('python') || lowerMsg.includes('โค้ด') || lowerMsg.includes('bug') || lowerMsg.includes('เขียนโปรแกรม')) {
      category = 'CODE'
    } else if (lowerMsg.includes('image') || lowerMsg.includes('draw') || lowerMsg.includes('รูปภาพ') || lowerMsg.includes('วาด')) {
      category = 'IMAGE'
    }

    switch (category) {
      case 'CODE':
        modelUsed = 'openai/gpt-4o-mini'
        const completion = await openai.chat.completions.create({
          model: 'openai/gpt-4o-mini',
          messages: [{ role: 'user', content: userPrompt }],
        })
        replyText = completion.choices[0].message.content || ''
        break;

      case 'IMAGE':
        modelUsed = 'image-generator'
        replyText = "คำขอเจนภาพ: ระบบกำลังเตรียมส่งคำสั่งไปที่ AI เจนภาพ..."
        break;

      default:
        modelUsed = 'gemini/gemini-3.6-flash'
        const model = geminiAI.getGenerativeModel({ model: 'gemini-3.6-flash' })
        const result = await model.generateContent(userPrompt)
        replyText = result.response.text()
        break;
    }

    return c.json({
      routedTo: category,
      model_used: modelUsed,
      result: replyText
    })
  } catch (error: any) {
    return c.json({ error: error.message }, 500)
  }
})

const port = parseInt(process.env.PORT || '8080', 10);
console.log(`Server is running on port ${port}`);

serve({
  fetch: app.fetch,
  port: port
})