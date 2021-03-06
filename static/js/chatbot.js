let botui = new BotUI('hello-world');

function chat_first(seq) {
  return botui.message.bot({
    delay: 2000, // delay时间过短，图片就不显示; loading=true时必须设置delay
    loading: true,
    photo: '../static/images/robot.jpg',   //以所在html文件位置为准
    content: '系统模型加载中，请稍后'
  })
  .then(function() {
    return botui.message.bot({
      delay: 60000, // delay时间过短，图片就不显示; loading=true时必须设置delay
      loading: true,
      photo: '../static/images/robot.jpg',   //以所在html文件位置为准
      content: seq
    })
  })
  .then(function () {
    return botui.action.text({
      delay: 200,
      addMessage:false, //不显示结果
      action: {
        value: '说点什么',  //预填充信息
        placeholder: '说点什么' //提示信息
      }
    })
  })
  .then(function (humanWords){
    botui.message.human({
      delay: 100,
      loading: true,
      photo: '../static/images/face3.jpg',
      content: humanWords.value
    });
    return humanWords.value
  })
  .then(function (humanWords){
    var bertResp = "nothing";
    $.ajax({
      url:"/chat", 
      type: "POST", 
      data: humanWords,
      dataType: "text",
      async : false, //同步执行
      success: function (msg) { 
        console.log(msg);
        bertResp = msg;
      }
    })
    console.log("outer " + bertResp);
    return bertResp
  })
  .then(function(bertResp){
    chat(bertResp);
  })
  .catch(error => console.log(error));
}

function chat(seq) {
  return botui.message.bot({
    delay: 200, // delay时间过短，图片就不显示; loading=true时必须设置delay
    loading: true,
    photo: '../static/images/robot.jpg',   //以所在html文件位置为准
    // content: 'Hello World from bot!'
    content: seq
  })
  .then(function () {
    return botui.action.text({
      delay: 200,
      addMessage:false, //不显示结果
      action: {
        value: '说点什么',  //预填充信息
        placeholder: '说点什么' //提示信息
      }
    })
  })
  .then(function (humanWords){
    botui.message.human({
      delay: 100,
      loading: true,
      photo: '../static/images/face.jpg',
      content: humanWords.value
    });
    return humanWords.value
  })
  .then(function (humanWords){
    var bertResp = "nothing";
    $.ajax({
      url:"/chat", 
      type: "POST", 
      data: humanWords,
      dataType: "text",
      async : false, //同步执行
      success: function (msg) { 
        console.log(msg);
        bertResp = msg;
      }
    })
    console.log("outer " + bertResp);
    return bertResp
  })
  .then(function(bertResp){
    chat(bertResp);
    return bertResp
  })
  .catch(error => console.log(error));
}


function show_backend() {
  return botui.message.bot({
    delay: 200, // delay时间过短，图片就不显示; loading=true时必须设置delay
    loading: true,
    content: '后台信息记录：'
  })
  .then(function (){
    var bertResp = "nothing";
    $.ajax({
      url:"/show_backend", 
      type: "POST", 
      // data: humanWords,
      dataType: "json",
      async : false, //同步执行
      success: function (msg) { 
        // console.log(msg);
        bertResp = msg;
      }
    })
    // bertResp = "历史抽取相似度： " + bertResp.scores 
    //         + "\r\n" + bertResp.history_input
    //         + "\r\n" + bertResp.kg_input 
    //         + "\r\n 候选回复语句： " + bertResp.candidate_response
    return bertResp
  })
  .then(function (bertResp){
    botui.message.bot({
      delay: 100,
      loading: true,
      content: "历史抽取相似度： " + bertResp.scores 
    });
    return bertResp
  })
  .then(function (bertResp){
    botui.message.bot({
      delay: 100,
      loading: true,
      content: bertResp.history_input + "\n"
    });
    return bertResp
  })
  .then(function (bertResp){
    botui.message.bot({
      delay: 100,
      loading: true,
      content: bertResp.kg_input 
    });
    return bertResp
  })
  .then(function (bertResp){
    botui.message.bot({
      delay: 100,
      loading: true,
      content: "候选回复语句： " + bertResp.candidate_response
    });
    return bertResp
  })
  .then(function () {
    return botui.action.text({
      delay: 200,
      addMessage:false, //不显示结果
      action: {
        value: '展示下一条信息',  //预填充信息
        placeholder: '展示下一条信息' //提示信息
      }
    })
  })
  .then(function(){
    show_backend();
  })
  .catch(error => console.log(error));
}

// 'scores_w': None,
// 'scores_s': None,
// 'scores' :None,
// 'history_input': None,
// 'kg_input': None,
// 'candidate_response': [],
// 'response': None