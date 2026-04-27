import plotly.graph_objects as go

# 1. 定义节点 (Nodes)
# 这里的顺序对应图中从左到右的标签
label = ["<18", "18-35", "35-65", ">65",           # Age (0-3)
         "S1", "S2", "S3",                         # Treatment (4-6)
         "Yes", "No",                              # Surgery (7-8)
         "Improve", "Worsen", "Stall"]             # Result (9-11)

# 2. 定义流向 (Links)
# source: 源节点索引, target: 目标节点索引, value: 流的力量（宽度）
source = [0, 0, 1, 1, 2, 2, 3, 4, 5, 5, 6, 7, 7, 8]
target = [4, 5, 5, 6, 4, 6, 6, 8, 7, 8, 7, 9, 10, 11]
value  = [10, 20, 15, 25, 10, 30, 15, 20, 25, 10, 40, 50, 10, 15]

# 3. 配置颜色 (可选，Plotly会自动分配，也可以手动指定)
color_node = ["#800000", "#800000", "#800000", "#800000", # Age组深红色
              "#D2B48C", "#D2B48C", "#D2B48C",             # Treatment组褐色
              "#F5DEB3", "#F5DEB3",                       # Surgery组浅色
              "#9ACD32", "#9ACD32", "#9ACD32"]             # Result组绿色

# 4. 创建桑基图对象
fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,           # 节点之间的垂直间距
      thickness = 20,     # 节点的宽度
      line = dict(color = "black", width = 0.5),
      label = label,
      color = color_node
    ),
    link = dict(
      source = source,
      target = target,
      value = value,
      # 设置透明度，模仿图中半透明流向效果
      color = "rgba(210, 180, 140, 0.4)"
  ))])

# 5. 设置布局和标题
fig.update_layout(title_text="Medical Treatment Pathway Analysis", font_size=12)
fig.show()

# 美赛保存图片技巧：
# fig.write_image("sankey.pdf") # 需要安装 kaleido 库