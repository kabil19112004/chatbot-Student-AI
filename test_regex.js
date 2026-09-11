const text = `
### Personalized 3-Day Academic Plan

1. **Day 1: Machine Learning Core**
   - Review Supervised vs Unsupervised models (45m)
   - Practice 10 quiz questions

2. **Day 2: Python Code Implementations**
   - Build a sample classification script (60m)
   - Review notes on functions and lambda
`;

let safe = text;
safe = safe.replace(/^\s*-\s+(.*$)/gim, '<li style="margin-left:18px;margin-bottom:4px;">$1</li>');
safe = safe.replace(/(?:<li[^>]*>.*?<\/li>\s*)+/g, match => `<ul style="margin:8px 0;">\n${match}</ul>\n`);

console.log(safe);
