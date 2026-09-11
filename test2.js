let safe = "abc\n<pre><code>line1\nline2</code></pre>\ndef";
console.log(safe.indexOf("\n"));
console.log(JSON.stringify(safe));
