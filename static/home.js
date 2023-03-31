function confirmLogout() {
    if (confirm('Are you really want to LOGOUT?')) {
    window.location.href = '/logout'; // Navigate to the Logout page
  }
}

function validate() {
  const option1 = document.querySelectorAll('input[name="region"]:checked');
  const option2 = document.querySelectorAll('input[name="user_type"]:checked');
  if (option1.length < 1) {
    alert('Please select at least one option for region.');
    return false;
  }
  if (option2.length < 1) {
    alert('Please select at least one option for user_type.');
    return false;
  }
  return true;
}

function confirmNavigation() {
    if (confirm('All the data will lost, are you sure you want to go back to the Home page?')) {
    window.location.href = '/home'; // Navigate to the Home page
}
}	 

