import { defineCollection, z } from 'astro:content'
import { glob } from 'astro/loaders'
import { categoryIds } from './config/categories'

const blog = defineCollection({
  loader: glob({ pattern: '**/*.{md,mdx}', base: './src/data/blog' }),
  schema: z.object({
    title: z.string(),
    description: z.string(),
    pubDate: z.coerce.date(),
    updatedDate: z.coerce.date().optional(),
    category: z.enum(categoryIds),
    tags: z.array(z.string()).default([]),
    draft: z.boolean().default(false)
  })
})

export const collections = { blog }
