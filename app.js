window.onload = onPageLoad;
function onPageLoad(){
    console.log("document loaded");
    var url = "http://127.0.0.1:5000/load_artifacts"
    $.get(url,function(data,status){
        if(data){
            console.log(data)
        }
    })
}

function sendResponse(rawText) {
    var url = "http://127.0.0.1:5000/get_response"
    $.post(url,{
        user_text:rawText
    },function(data, status){
        if(data !== "undefined"){
            console.log(data['bot_response'])
            var botHtml = '<p class="botText"><span>' + data['bot_response'] + '</span></p>';
            $("#chatbox").append(botHtml);
            document.getElementById('userInput').scrollIntoView({block: 'start', behavior: 'smooth'});
            return data['bot_response']
            
        }
    })
  }