(function() {
  var style = getComputedStyle(document.documentElement);
  var accent = style.getPropertyValue('--accent').trim();
  var accent2 = style.getPropertyValue('--accent2').trim();
  var ink = style.getPropertyValue('--ink').trim();
  var muted = style.getPropertyValue('--muted').trim();
  var rule = style.getPropertyValue('--rule').trim();
  var bg2 = style.getPropertyValue('--bg2').trim();
  var success = style.getPropertyValue('--success').trim();
  var warning = style.getPropertyValue('--warning').trim();
  var danger = style.getPropertyValue('--danger').trim();

  // --- Chart: Benefit vs Difficulty Matrix ---
  var chartMatrix = echarts.init(document.getElementById('chart-matrix'), null, { renderer: 'svg' });

  var directions = [
    { name: 'K量化进阶 (softmax零空间)', benefit: 25, difficulty: 30, priority: 'high' },
    { name: 'ADMM联合优化', benefit: 20, difficulty: 40, priority: 'high' },
    { name: 'KFAC Hessian', benefit: 22, difficulty: 50, priority: 'high' },
    { name: '可学习蝴蝶变换', benefit: 15, difficulty: 60, priority: 'medium' },
    { name: '闭式联合尺度优化', benefit: 12, difficulty: 35, priority: 'medium' },
    { name: 'V量化专项优化', benefit: 15, difficulty: 20, priority: 'medium' },
    { name: 'Attention-aware Hessian', benefit: 12, difficulty: 65, priority: 'medium' },
    { name: '跨块联合优化', benefit: 10, difficulty: 70, priority: 'medium' },
    { name: '混合精度探索', benefit: 8, difficulty: 75, priority: 'low' },
    { name: '晶格码本/VQ', benefit: 10, difficulty: 85, priority: 'low' },
  ];

  function getPriorityColor(p) {
    if (p === 'high') return success;
    if (p === 'medium') return warning;
    return accent;
  }

  function getPriorityLabel(p) {
    if (p === 'high') return '最高优先级';
    if (p === 'medium') return '中高优先级';
    return '较低优先级';
  }

  var scatterData = directions.map(function(d, i) {
    return {
      value: [d.difficulty, d.benefit, d.name],
      itemStyle: { color: getPriorityColor(d.priority) },
      name: d.name,
      priority: d.priority
    };
  });

  chartMatrix.setOption({
    animation: false,
    tooltip: {
      appendToBody: true,
      formatter: function(params) {
        return '<strong>' + params.data.name + '</strong><br/>' +
               '预期收益: ' + params.data.value[1] + '%<br/>' +
               '实施难度: ' + params.data.value[0] + '/100<br/>' +
               '优先级: ' + getPriorityLabel(params.data.priority);
      }
    },
    grid: {
      left: '8%',
      right: '5%',
      top: '15%',
      bottom: '12%'
    },
    xAxis: {
      name: '实施难度 →',
      nameTextStyle: { color: muted, fontSize: 12 },
      min: 10,
      max: 95,
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    yAxis: {
      name: '预期收益 (%)',
      nameTextStyle: { color: muted, fontSize: 12 },
      min: 0,
      max: 30,
      axisLine: { lineStyle: { color: rule } },
      axisTick: { show: false },
      axisLabel: { color: muted, fontSize: 11 },
      splitLine: { lineStyle: { color: rule, type: 'dashed' } }
    },
    series: [{
      type: 'scatter',
      data: scatterData,
      symbolSize: function(data) {
        var base = 14;
        if (data.priority === 'high') return base + 6;
        if (data.priority === 'medium') return base + 2;
        return base;
      },
      label: {
        show: true,
        formatter: function(params) {
          var names = params.data.name.split(' ');
          return names[0];
        },
        position: 'top',
        fontSize: 10,
        color: ink,
        fontWeight: 500
      },
      itemStyle: {
        borderWidth: 2,
        borderColor: bg2,
        shadowBlur: 4,
        shadowColor: 'rgba(0,0,0,0.1)'
      },
      markArea: {
        silent: true,
        itemStyle: {
          opacity: 0.08
        },
        data: [
          [
            { xAxis: 10, yAxis: 15, itemStyle: { color: success } },
            { xAxis: 45, yAxis: 30 }
          ]
        ]
      }
    }],
    legend: {
      data: [
        { name: '最高优先级', itemStyle: { color: success } },
        { name: '中高优先级', itemStyle: { color: warning } },
        { name: '较低优先级', itemStyle: { color: accent } }
      ],
      top: 5,
      textStyle: { color: muted, fontSize: 11 },
      itemWidth: 10,
      itemHeight: 10
    }
  });

  window.addEventListener('resize', function() { chartMatrix.resize(); });
})();
