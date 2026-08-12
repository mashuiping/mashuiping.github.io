import { defineConfig } from 'astro/config'
import sitemap from '@astrojs/sitemap'
import { unified } from '@astrojs/markdown-remark'
import remarkMath from 'remark-math'
import rehypeKatex from 'rehype-katex'

export default defineConfig({
  site: 'https://mashuiping.github.io',
  output: 'static',
  trailingSlash: 'always',
  integrations: [sitemap()],
  markdown: {
    // Astro 7 默认是 Sätteri；要用 remark/rehype（含 KaTeX）需显式切回 unified
    processor: unified({
      remarkPlugins: [remarkMath],
      rehypePlugins: [rehypeKatex]
    }),
    syntaxHighlight: {
      type: 'shiki',
      excludeLangs: ['math', 'mermaid']
    },
    shikiConfig: {
      theme: 'github-dark',
      wrap: true
    }
  }
})
