var myChart = echarts.init(document.querySelector('.box'));
var maxSeriesLength = 20;
var data = [];

$(function (){
    // 绑定点击 PredictBatchSize 事件
    PredictWindowSize();
    // 获取补全模型列表
    ImputeModelSelect();
    //获取预测模型列表
    PredictModelSelect();
    // Task
    Task();
    // 保存任务配置
    TaskSetSave();
    // 初始化缺失率饼图
    initMissingRateChart();
    // 初始化异常率饼图
    initAnomalyRateChart();
    // 获取表格数据
    ShowTaskResults();
    // 点击 details 按钮
    bindBtnDetails();
    // 保存 details 设置按钮
    bindBtnSaveDetails();
})

function PredictWindowSize(){
    $('#PredictWindowSize a').click(function(){
        $('#BtnPredictWindowSize').html($(this).text() + ' <span class="caret"></span>');
    });
}

function ImputeModelSelect() {
    var data = {
        models: ["Model 1", "Model 2", "Model 3", "Model 4"]
    };

    var selectList = $('#ImputeModelSelect');

    $.each(data.models, function(i, model) {
        var listItem = $('<li><label><input type="radio" name="models" value="' + model + '">' + model + '</label></li>');
        listItem.on('click', function() {
            var selectedModel = $(this).text();
            $('#ImputeModel').text(selectedModel);
        });
        selectList.append(listItem);
    });
}

function PredictModelSelect() {
    var data = {
        models: ["Model 1", "Model 2", "Model 3", "Model 4"]
    };

    var selectList = $('#PredictModelSelect');

    $.each(data.models, function(i, model) {
        var listItem = $('<li><label><input type="radio" name="models" value="' + model + '">' + model + '</label></li>');
        listItem.on('click', function() {
            var selectedModel = $(this).text();
            $('#PredictModel').text(selectedModel);
        });
        selectList.append(listItem);
    });
}

function Task() {
    let intervalId;
    let socket = new WebSocket("ws://localhost:8000/ws/task/")
    var chartControl_missing = initMissingRateChart();

    socket.onopen = function (e){
        console.log("Connection open");
    };

    document.getElementById('taskToggle').addEventListener('change', (event) => {
      if (event.target.checked) {
        document.getElementById('click-trigger').className = 'glyphicon glyphicon-record processing-color';
        socket.send(JSON.stringify({"type": "task.start"}));
      } else {
        socket.send(JSON.stringify({"type": "task.stop"}));
        document.getElementById('click-trigger').className = 'glyphicon glyphicon-record default-color';
        document.getElementById('predict').className = 'glyphicon glyphicon-record default-color';
        document.getElementById('finished').className = 'glyphicon glyphicon-record default-color';
        document.getElementById("Status").textContent = "Stopped"
        clearInterval(intervalId);
      }
    });

    socket.onmessage = function(event){
        console.log(`return message:${event.data}`);
        let data = JSON.parse(event.data);

        if(data.hasOwnProperty("impute_data")) {
        // 更新饼图
        chartControl_missing.updateChart(data.impute_data);
    }

        if (data.status === "progressing") {
            document.getElementById('predict').className = 'glyphicon glyphicon-record processing-color';
        } else if (data.status === "finished") {
            document.getElementById('finished').className = 'glyphicon glyphicon-record processing-color';
            clearInterval(intervalId);
        }

        if(data.hasOwnProperty("impute_data")) {
             // 更新图表数据
             chartControl_missing.updateChart(data.impute_data);
        }

        let start_time = new Date(data.start_time * 1000);
        document.getElementById("Status").textContent = data.status;

        if (intervalId) {
            clearInterval(intervalId);
        }
        intervalId = setInterval(() => {
            updateTaskTime(start_time);
        }, 1000);

    }

    function updateTaskTime(StartTime) {
    let currentTime = new Date();
    if (document.getElementById("Status").textContent !== "finished") {
        let Time = parseInt((currentTime - StartTime) / 1000);
        let formattedImputeTime = formatTime(Time);
        document.getElementById("TaskTime").textContent = formattedImputeTime;
    }

}

// 将获得的秒数转换为 HH:MM:SS 格式
    function formatTime(seconds) {
        let hours = parseInt(seconds / 3600);
        let minutes = parseInt((seconds % 3600) / 60);
        let remainingSeconds = seconds % 60;

        return `${hours}:${minutes}:${remainingSeconds}`;
    }

    socket.onclose = function(event) {
        if (event.wasClean) {
            console.log(`Connection closed properly：${event.code},${event.reason}`);
        } else {
            console.log('disconnect');
        }
        clearInterval(intervalId);
    };

    socket.onerror = function(error) {
        console.log(`[error] ${error.message}`);
    };
}

function TaskSetSave() {
  document.getElementById('TaskSaveToggle').addEventListener('click', (event) => {
    const fetchParams = () => ({
      ImputeModel: $('#ImputeModel').text(),
      PredictModel: $('#PredictModel').text(),
      PredictWindowSize: $('#BtnPredictWindowSize').text().trim()
    });

    const params = fetchParams();

    $.ajax({
      type: "POST",
      url: "/task/save/",
      contentType: "application/json",
      dataType: "JSON",
      data: JSON.stringify(params),
      success: function (data) {
          console.log(JSON.stringify(params));
          console.log("Success: ", data);
      },
      error: function (error) {
          console.log(error);
      }
    });
  });
}

function initChart(){
    option = {

        title: {
            text: 'Dynamic Data'
        },

        tooltip: {
            trigger: 'axis',
            formatter: function (params) {
                const date = params[0].name.split("_")[0];
                return (
                    date +
                    ' : ' +
                    params[0].value[1]
                );
            },
            axisPointer: {
                animation: false
            }
        },

        xAxis: {
            type: 'category',
            splitLine: {
                show: false
            }
        },
        yAxis: {
            type: 'value',
            boundaryGap: [0, '100%'],
            splitLine: {
                show: false
            }
        },
        series: []
    };
    myChart.setOption(option);
}


function initMissingRateChart(){
var myChart2 = echarts.init(document.getElementById('missing_rate_chart'));
    option = {
  title: {
    text: 'missing_rate_chart',
    left: 'center',
    top: '18%'
  },
  tooltip: {
    trigger: 'item'
  },
  legend: {
       top: '5%',
  },
  series: [
    {
      name: 'Access From',
      type: 'pie',
      radius: '50%',
      data: [
      ],
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      },
    }
  ]
};
    myChart2.setOption(option);
    // 在这里定义你的updateChart函数
    function updateChart(data) {
        // 将数据转换为 ECharts 饼图所接受的格式
        var pieData = Object.entries(data).map(([key, value]) => ({name: key, value: value}));

        // 使用 setOption 更新饼图数据
        myChart2.setOption({
            series: [{
                data: pieData
            }]
        });
    }
    return {
        updateChart: updateChart
    };
}

function initAnomalyRateChart(){
    var myChart3 = echarts.init(document.getElementById('anomaly_rate_chart'));
    option = {
  title: {
    text: 'anomaly_rate_chart',
    left: 'center',
    top: '18%'
  },
  tooltip: {
    trigger: 'item'
  },
  legend: {
       top: '5%',
  },
  series: [
    {
      name: 'Access From',
      type: 'pie',
      radius: '50%',
      data: [
        { value: 1048, name: 'Search Engine' },
        { value: 735, name: 'Direct' },
        { value: 580, name: 'Email' },
        { value: 484, name: 'Union Ads' },
        { value: 300, name: 'Video Ads' }
      ],
      emphasis: {
        itemStyle: {
          shadowBlur: 10,
          shadowOffsetX: 0,
          shadowColor: 'rgba(0, 0, 0, 0.5)'
        }
      },
    }
  ]
};
    myChart3.setOption(option);
}

function ShowTaskResults(){
    $(document).on('click', '.pagination1 a', function (e) {
        e.preventDefault();
        var page = $(this).data('page');
        $.ajax({
            url: '/load_impute_results/',  // Set this URL to load impute results data
            data: {'page': page},
            method: 'GET',
            success: function (data) {
                $('#impute-results-table').html(data.html);  // Update the impute results table
            },
            error: function (xhr, ajaxOptions, thrownError) {
                console.log(thrownError);
            }
        });
    });

    $(document).on('click', '.pagination2 a', function (e) {  // Add a new handler for the second table
        e.preventDefault();

        var page = $(this).data('page');

        $.ajax({
            url: '/load_anomaly_results/',  // Set this URL to load anomaly results data
            data: {'page': page},
            method: 'GET',
            success: function (data) {
                $('#Anomaly-results-table').html(data.html);  // Update the anomaly results table
            },
            error: function (xhr, ajaxOptions, thrownError) {
                console.log(thrownError);
            }
        });
    });

    $('#impute-results-table').on('submit', 'form', function(e) {
        e.preventDefault();
        var page = $("input[name='page']", this).val();

        $.ajax({
            url: '/load_impute_results/',  // Set this URL to load impute results data
            type: 'GET',
            data: {'page': page},
            success: function (data) {
                $('#impute-results-table').html(data.html);
            }
        });
    });

    $('#Anomaly-results-table').on('submit', 'form', function(e) {
        e.preventDefault();
        var page = $("input[name='page']", this).val();

        $.ajax({
            url: '/load_anomaly_results/',  // Set this URL to load anomaly results data
            type: 'GET',
            data: {'page': page},
            success: function (data) {
                $('#Anomaly-results-table').html(data.html);
            }
        });
    });
}

function bindBtnDetails() {
    $(".btn-edit").click(function () {
        //清空对话框中的数据
        $('#formAdd')[0].reset();
        var currentId = $(this).attr('uid');
        EDIT_ID = currentId;
        $("#myModal").modal('show');
        $.ajax({
            url: "/get/analysis/",
            type: 'GET',
            data: {uid: currentId},
            dataType: "JSON",
            success: function (data) {
                if(data.status == 'ERROR'){
                    alert('Error: ' + data.error);
                } else {
                    $("#analysisInput").val(data.analysis);
                }
            },
            error: function (jqXHR, textStatus, errorThrown) {
                //服务器连接失败时的处理
                alert('Failed to connect to server: ' + textStatus);
            }
        });
    })
}

function bindBtnSaveDetails(){
    $("#btnSave").click(function () {
    $.ajax({
        url: "/save/analysis/",
        type: 'POST',
        data: {
            uid: EDIT_ID,
            analysis: $("#analysisInput").val()
        },
        dataType: "JSON",
        success: function (data) {
            if(data.status == 'OK'){
                alert('Saved successfully');
                $("#myModal").modal('hide'); // 关闭模态框
                // 你可以在这里添加代码来更新页面的其他元素，例如列表或者数据表格，来显示新的数据
            } else {
                alert('Failed to save: ' + data.error);
            }
        },
        error: function (jqXHR, textStatus, errorThrown) {
            alert('Failed to connect to server: ' + textStatus);
        }
    });
})
}