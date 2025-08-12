document.addEventListener('DOMContentLoaded', () => {
    const crowd1 = document.querySelector('.crowd');
    const crowd2 = document.querySelector('.crowd2');

    function getRandomEmoji() {
        const start = 0x1F600;
        const end = 0x1F64F;
        const codePoint = Math.floor(Math.random() * (end - start + 1)) + start;
        return String.fromCodePoint(codePoint);
    }

    const numEmojis = 500; 

    let crowd1Content = '';
    let crowd2Content = '';

    for (let i = 0; i < numEmojis; i++) {
        crowd1Content += getRandomEmoji();
        crowd2Content += getRandomEmoji();
    }
    
    crowd1.textContent = crowd1Content;
    crowd2.textContent = crowd2Content;
});
