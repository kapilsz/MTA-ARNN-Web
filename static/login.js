function redirectToRequestPage() {
    window.location.href = '/requestt'; // Navigate to the request page
}

function redirectToHomePage() {
  window.location.href = '/'; // Navigate to the request page
}

function backToHome() {
      window.location.href = '/'; // Navigate to the request page
      // redirect to login page with message as URL parameter
      // const message = encodeURIComponent('Your request has been sent.');
      // window.location.href = `/?message=${message}`;
  }

function showAccessRequestForm() {
  document.querySelector('.access-request-form').style.display = 'block';
  document.querySelector('.access-request-message').style.display = 'none';
}

function cancelAccessRequest() {
  document.querySelector('.access-request-form').style.display = 'none';
}

function submitAccessRequest(event) {
  event.preventDefault();
  // Perform form submission logic here (e.g. send data to server)
  // ...

  // Hide the access request form and show the message
  document.querySelector('.access-request-form').style.display = 'none';
  document.querySelector('.access-request-message').style.display = 'block';
}

function backToLogin() {
  window.location.href = '/';
}
