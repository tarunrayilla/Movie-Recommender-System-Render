form = document.getElementById('myForm')
var msg = document.getElementById('message')

form.addEventListener('submit', (event) => {
    let rate = form.rating
    console.log(rate)
    console.log(rate.value)
    console.log(typeof(rate.value))
    console.log('hello')
    if(rate.value == '') {
        event.preventDefault()
        msg.innerHTML = 'Please give a valid rating'
    }
})
