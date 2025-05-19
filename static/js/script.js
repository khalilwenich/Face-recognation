// Common JavaScript functions for Face Recognition System

// Auto-dismiss alerts after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    // Get all alert elements with the 'alert-dismissible' class
    const alerts = document.querySelectorAll('.alert-dismissible');
    
    // Set a timeout to close each alert after 5 seconds
    alerts.forEach(function(alert) {
        setTimeout(function() {
            // Create a bootstrap alert instance and close it
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});
