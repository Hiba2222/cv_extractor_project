// Initialize when DOM is fully loaded
document.addEventListener('DOMContentLoaded', function() {
    // Upload page elements
    const dropZone = document.getElementById('dropZone');
    const fileInput = document.getElementById('pdf_file');
    const browseButton = document.getElementById('browseButton');
    const fileInfo = document.getElementById('fileInfo');
    const fileName = document.getElementById('fileName');
    const submitBtn = document.getElementById('submitBtn');
    const modelCards = document.querySelectorAll('.model-card');
    
    // Results page elements
    const modelSelectors = document.querySelectorAll('input[name="model-selector"]');
    
    // === Upload Page Functions ===
    
    // Handle browse button click
    if (browseButton) {
        browseButton.addEventListener('click', function() {
            fileInput.click();
        });
    }
    
    // Handle file selection
    if (fileInput) {
        fileInput.addEventListener('change', function() {
            if (fileInput.files.length > 0) {
                fileInfo.classList.remove('d-none');
                fileName.textContent = fileInput.files[0].name;
                submitBtn.classList.remove('disabled');
            } else {
                fileInfo.classList.add('d-none');
                submitBtn.classList.add('disabled');
            }
        });
    }
    
    // Handle drag and drop
    if (dropZone) {
        // Prevent default drag behaviors
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, preventDefaults, false);
        });
        
        // Highlight drop zone when file is dragged over it
        ['dragenter', 'dragover'].forEach(eventName => {
            dropZone.addEventListener(eventName, highlight, false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            dropZone.addEventListener(eventName, unhighlight, false);
        });
        
        // Handle dropped files
        dropZone.addEventListener('drop', handleDrop, false);
    }
    
    // Handle model selection
    if (modelCards.length > 0) {
        modelCards.forEach(card => {
            card.addEventListener('click', function() {
                // Remove selected class from all cards
                modelCards.forEach(c => c.classList.remove('selected'));
                
                // Add selected class to clicked card
                this.classList.add('selected');
                
                // Select the radio button
                const model = this.getAttribute('data-model');
                document.querySelector(`input[value="${model}"]`).checked = true;
            });
        });
        
        // Set first model as selected by default
        modelCards[0].classList.add('selected');
    }
    
    // === Results Page Functions ===
    
    // Handle model selector tabs in results page
    if (modelSelectors.length > 0) {
        modelSelectors.forEach(selector => {
            selector.addEventListener('change', function() {
                // Hide all result divs
                const resultDivs = document.querySelectorAll('.model-result');
                resultDivs.forEach(div => div.style.display = 'none');
                
                // Show the selected model's results
                const modelName = this.id.replace('-btn', '');
                document.getElementById(modelName + '-results').style.display = 'block';
            });
        });
    }
    
    // === Helper Functions ===
    
    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }
    
    function highlight() {
        dropZone.classList.add('highlight');
    }
    
    function unhighlight() {
        dropZone.classList.remove('highlight');
    }
    
    function handleDrop(e) {
        if (e.dataTransfer.files.length > 0) {
            const file = e.dataTransfer.files[0];
            if (file.type === 'application/pdf') {
                // Create a DataTransfer object to update the FileList
                const dataTransfer = new DataTransfer();
                dataTransfer.items.add(file);
                fileInput.files = dataTransfer.files;
                
                fileInfo.classList.remove('d-none');
                fileName.textContent = file.name;
                submitBtn.classList.remove('disabled');
            } else {
                showAlert('Please upload a PDF file', 'danger');
            }
        }
    }
    
    function showAlert(message, type = 'warning') {
        const alertDiv = document.createElement('div');
        alertDiv.className = `alert alert-${type} alert-dismissible fade show`;
        alertDiv.innerHTML = `
            ${message}
            <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
        `;
        
        const container = document.querySelector('.container') || document.body;
        container.insertBefore(alertDiv, container.firstChild);
        
        // Auto dismiss after 5 seconds
        setTimeout(() => {
            alertDiv.classList.remove('show');
            setTimeout(() => alertDiv.remove(), 500);
        }, 5000);
    }
}); 