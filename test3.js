let safe = "abc\n<pre><code>line1\nline2</code></pre>\ndef";
let parts = safe.split(/(<pre><code>[\s\S]*?<\/code><\/pre>)/gi);
for (let i = 0; i < parts.length; i++) {
  if (!parts[i].startsWith("<pre>")) {
    parts[i] = parts[i].replace(/\n/g, "<br>");
  }
}
let res = parts.join("");
console.log(JSON.stringify(res));
