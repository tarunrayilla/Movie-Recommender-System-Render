var check_option_error = document.getElementById('check_option_error')


function handleData() {
    var form_data = new FormData(document.querySelector("#regForm"));
    if(!form_data.has("genres")) {
        check_option_error.style.visibility = "visible";
        return false;
    }
    else {
        check_option_error.style.visibility = "hidden";
        console.log("hello")
    }
    return true;
    
}