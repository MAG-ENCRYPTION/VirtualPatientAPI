fetch('http://127.0.0.1:8000/api/tutor-module/response/', {
    method: 'POST',
    headers: {
        'Content-Type': 'application/json',
    },
    body: JSON.stringify({
        question: "Quel est le problème de santé ?",
        clinical_case: 123
    }),
})
.then(response => response.json())
.then(data => console.log(data))
.catch(error => console.error('Error:', error));
