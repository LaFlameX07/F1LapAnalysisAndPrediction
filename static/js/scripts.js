document.addEventListener('DOMContentLoaded', function() {
    // Add interactive elements if needed
    const links = document.querySelectorAll('nav ul li a');

    links.forEach(link => {
        link.addEventListener('mouseover', () => {
            link.style.color = '#ff4500';
        });
        link.addEventListener('mouseout', () => {
            link.style.color = '#fff';
        });
    });
});
