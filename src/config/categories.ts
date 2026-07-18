export const categoryIds = ['ai-infra', 'k8s', 'golang', 'investing', 'notes'] as const

export type CategoryId = (typeof categoryIds)[number]

export const categories: Record<
  CategoryId,
  { label: string; description: string }
> = {
  'ai-infra': {
    label: 'AI Infra',
    description: 'AI 基础设施、推理服务与算力系统'
  },
  k8s: {
    label: 'K8s',
    description: 'Kubernetes、云原生平台与集群实践'
  },
  golang: {
    label: 'Golang',
    description: 'Go 工程实践、语言设计与性能优化'
  },
  investing: {
    label: '投资',
    description: '行业研究、公司分析与投资思考'
  },
  notes: {
    label: '杂记',
    description: '阅读、工具、生活与未归入其他主题的记录'
  }
}
