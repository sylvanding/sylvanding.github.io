let s1 = '2024-5-28'; //website start date
s1 = new Date(s1.replace(/-/g, "/"));
let s2 = new Date();
let timeDifference = s2.getTime() - s1.getTime();

let days = Math.floor(timeDifference / (1000 * 60 * 60 * 24));

let result = days + "";
document.getElementById('runningdays').innerHTML = result;