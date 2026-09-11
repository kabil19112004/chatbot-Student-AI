const escapeHTML = (str) => str.replace(/[&<>'"]/g, tag => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'
  }[tag]));

function formatMarkdown(text) {
  if (!text) return "";
  let safe = escapeHTML(text);
  safe = safe.replace(/```([a-z0-9_-]*)\n([\s\S]*?)```/gi, '<pre><code>$2</code></pre>');
  safe = safe.replace(/`([^`]+)`/g, '<code>$1</code>');
  safe = safe.replace(/^### (.*$)/gim, '<h3 style="font-size:16px;font-weight:700;margin:12px 0 6px;">$1</h3>');
  safe = safe.replace(/^## (.*$)/gim, '<h2 style="font-size:18px;font-weight:700;margin:14px 0 8px;">$1</h2>');
  safe = safe.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  safe = safe.replace(/\*([^*]+)\*/g, '<em>$1</em>');
  safe = safe.replace(/^\s*-\s+(.*$)/gim, '<li style="margin-left:18px;margin-bottom:4px;">$1</li>');
  safe = safe.replace(/(?:<li[^>]*>.*?<\/li>\s*)+/g, match => '<ul style="margin:8px 0;">' + match + '</ul>');
  
  // Newline replacement - should exclude inside <pre>
  // A simple way is to split by <pre> and only replace \n outside of it.
  
  let parts = safe.split(/(<pre><code>[\s\S]*?<\/code><\/pre>)/gi);
  for (let i = 0; i < parts.length; i++) {
    if (!parts[i].startsWith('<pre>')) {
      parts[i] = parts[i].replace(/\n/g, "<br>");
    }
  }
  safe = parts.join("");
  return safe;
}

const sample = `Here is some code:
\`\`\`python
def hello():
    print("hello")
    return True
\`\`\`
Hope it helps!`;

console.log(formatMarkdown(sample));
