$(document).ready(function(){
  $("#predict").click(function(){
      const txt = $("#news_content").val(); // 获取新闻内容
    $.post("/predict", { news: txt }, function(result){ // 调用后端的接口，传递新闻内容
      $("#News_category").html(result);
    });
  });
});
