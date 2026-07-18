import rss from '@astrojs/rss'
import { getCollection } from 'astro:content'
import { categories } from '../config/categories'

export async function GET(context: { site: URL }) {
  const posts = (await getCollection('blog', ({ data }) => !data.draft))
    .sort((a, b) => b.data.pubDate.valueOf() - a.data.pubDate.valueOf())

  return rss({
    title: 'masp 的博客',
    description: '记录技术、思考与实践',
    site: context.site,
    customData: '<language>zh-CN</language>',
    items: posts.map((post) => ({
      title: post.data.title,
      description: post.data.description,
      pubDate: post.data.pubDate,
      categories: [categories[post.data.category].label, ...post.data.tags],
      link: `/blog/${post.id}/`
    }))
  })
}
