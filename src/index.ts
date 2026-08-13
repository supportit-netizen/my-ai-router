import 'dotenv/config'
import { Hono } from 'hono'
import { cors } from 'hono/cors'
import { serve } from '@hono/node-server'
import { GoogleGenerativeAI } from '@google/generative-ai'
import OpenAI from 'openai'
import { Pool } from 'pg'
import bcrypt from 'bcryptjs'
import jwt from 'jsonwebtoken'
// ⚠️ สำคัญ: import เข้า lib โดยตรง ห้ามใช้ 'pdf-parse' เฉย ๆ
// เพราะ index.js ของ pdf-parse มี debug code ที่อ่านไฟล์ ./test/data/05-versions-space.pdf
// เมื่อ bundle เป็นไฟล์เดียว (module.parent === null) มันจะเข้า debug mode แล้ว crash ENOENT
// @ts-ignore - subpath นี้ไม่มี type declaration
import pdfParse from 'pdf-parse/lib/pdf-parse.js'

const app = new Hono()
app.use('/*', cors())

const geminiAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY || '')
const openai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY || '',
  baseURL: 'https://openrouter.ai/api/v1',
})

const JWT_SECRET = process.env.JWT_SECRET || 'change_this_secret'

const pool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: process.env.DATABASE_URL?.includes('railway') ? { rejectUnauthorized: false } : undefined,
})

async function initDb() {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS users (
      id SERIAL PRIMARY KEY,
      email TEXT UNIQUE NOT NULL,
      password_hash TEXT NOT NULL,
      created_at TIMESTAMP DEFAULT NOW()
    )
  `)
}
initDb().catch((e) => console.error('DB init failed:', e.message))

app.get('/', (c) => {
  return c.text('AI Router Server is Ready!')
})

app.post('/api/register', async (c) => {
  try {
    const { email, password } = await c.req.json()
    if (!email || !password) {
      return c.json({ error: 'กรุณากรอกอีเมลและรหัสผ่าน' }, 400)
    }
    if (password.length < 6) {
      return c.json({ error: 'รหัสผ่านต้องมีอย่างน้อย 6 ตัวอักษร' }, 400)
    }

    const existing = await pool.query('SELECT id FROM users WHERE email = $1', [email])
    if (existing.rows.length > 0) {
      return c.json({ error: 'อีเมลนี้ถูกใช้งานแล้ว' }, 409)
    }

    const passwordHash = await bcrypt.hash(password, 10)
    const result = await pool.query(
      'INSERT INTO users (email, password_hash) VALUES ($1, $2) RETURNING id, email',
      [email, passwordHash]
    )

    const user = result.rows[0]
    const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET, { expiresIn: '7d' })

    return c.json({ token, email: user.email })
  } catch (error: any) {
    return c.json({ error: error.message }, 500)
  }
})

app.post('/api/login', async (c) => {
  try {
    const { email, password } = await c.req.json()
    if (!email || !password) {
      return c.json({ error: 'กรุณากรอกอีเมลและรหัสผ่าน' }, 400)
    }

    const result = await pool.query('SELECT id, email, password_hash FROM users WHERE email = $1', [email])
    if (result.rows.length === 0) {
      return c.json({ error: 'อีเมลหรือรหัสผ่านไม่ถูกต้อง' }, 401)
    }

    const user = result.rows[0]
    const valid = await bcrypt.compare(password, user.password_hash)
    if (!valid) {
      return c.json({ error: 'อีเมลหรือรหัสผ่านไม่ถูกต้อง' }, 401)
    }

    const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET, { expiresIn: '7d' })
    return c.json({ token, email: user.email })
  } catch (error: any) {
    return c.json({ error: error.message }, 500)
  }
})

async function requireAuth(c: any, next: any) {
  const authHeader = c.req.header('Authorization')
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return c.json({ error: 'กรุณาเข้าสู่ระบบ' }, 401)
  }
  const token = authHeader.replace('Bearer ', '')
  try {
    const payload = jwt.verify(token, JWT_SECRET) as any
    c.set('userId', payload.userId)
    c.set('userEmail', payload.email)
    await next()
  } catch {
    return c.json({ error: 'เซสชันหมดอายุ กรุณาเข้าสู่ระบบใหม่' }, 401)
  }
}

async function generateImage(prompt: string): Promise<string> {
  const res = await fetch('https://openrouter.ai/api/v1/images', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model: 'openai/gpt-image-1',
      prompt,
    }),
  })
  const data = await res.json()
  if (!res.ok) {
    throw new Error(data.error?.message || 'สร้างภาพไม่สำเร็จ')
  }
  const b64 = data.data?.[0]?.b64_json
  if (!b64) throw new Error('ไม่ได้รับข้อมูลภาพกลับมา')
  return `data:image/png;base64,${b64}`
}

app.post('/api/chat', requireAuth, async (c) => {
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
    let imageUrl: string | null = null

    if (lowerMsg.includes('code') || lowerMsg.includes('python') || lowerMsg.includes('โค้ด') || lowerMsg.includes('bug') || lowerMsg.includes('เขียนโปรแกรม')) {
      category = 'CODE'
    } else if (lowerMsg.includes('image') || lowerMsg.includes('draw') || lowerMsg.includes('รูปภาพ') || lowerMsg.includes('วาด')) {
      category = 'IMAGE'
    }

    switch (category) {
      case 'CODE': {
        modelUsed = 'openai/gpt-4o-mini'
        const completion = await openai.chat.completions.create({
          model: 'openai/gpt-4o-mini',
          messages: [{ role: 'user', content: userPrompt }],
        })
        replyText = completion.choices[0].message.content || ''
        break
      }

      case 'IMAGE': {
        modelUsed = 'openai/gpt-image-1'
        imageUrl = await generateImage(userPrompt)
        replyText = 'สร้างภาพเรียบร้อยแล้วครับ'
        break
      }

      default: {
        modelUsed = 'gemini/gemini-3.6-flash'
        const model = geminiAI.getGenerativeModel({ model: 'gemini-3.6-flash' })
        const result = await model.generateContent(userPrompt)
        replyText = result.response.text()
        break
      }
    }

    return c.json({
      routedTo: category,
      model_used: modelUsed,
      result: replyText,
      imageUrl,
    })
  } catch (error: any) {
    return c.json({ error: error.message }, 500)
  }
})

app.post('/api/upload', requireAuth, async (c) => {
  try {
    const body = await c.req.parseBody()
    const file = body['file'] as File | undefined

    if (!file) {
      return c.json({ error: 'กรุณาแนบไฟล์' }, 400)
    }

    const arrayBuffer = await file.arrayBuffer()
    const buffer = Buffer.from(arrayBuffer)
    let extractedText = ''

    if (file.name.toLowerCase().endsWith('.pdf')) {
      // ตรวจ magic number กัน buffer ที่ไม่ใช่ PDF จริง
      if (buffer.subarray(0, 4).toString('latin1') !== '%PDF') {
        return c.json({ error: 'ไฟล์นี้ไม่ใช่ PDF ที่ถูกต้อง' }, 400)
      }
      try {
        const parsed = await pdfParse(buffer)
        extractedText = parsed.text
      } catch (e: any) {
        return c.json({ error: `อ่าน PDF ไม่สำเร็จ: ${e.message}` }, 400)
      }
    } else {
      extractedText = buffer.toString('utf-8')
    }

    if (!extractedText.trim()) {
      return c.json({ error: 'ไม่สามารถอ่านเนื้อหาจากไฟล์นี้ได้' }, 400)
    }

    const truncated = extractedText.slice(0, 12000)

    const completion = await openai.chat.completions.create({
      model: 'openai/gpt-4o-mini',
      messages: [
        {
          role: 'user',
          content: `สรุปเนื้อหาไฟล์ต่อไปนี้เป็นภาษาไทย กระชับ ได้ใจความ:\n\n${truncated}`,
        },
      ],
    })

    const summary = completion.choices[0].message.content || ''

    return c.json({
      fileName: file.name,
      summary,
      model_used: 'openai/gpt-4o-mini',
    })
  } catch (error: any) {
    return c.json({ error: error.message }, 500)
  }
})

const port = parseInt(process.env.PORT || '8080', 10)
console.log(`Server is running on port ${port}`)

serve({
  fetch: app.fetch,
  port: port,
})