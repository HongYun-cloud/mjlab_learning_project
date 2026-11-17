---
name: Chaoxing Quiz Runner
description: 在学习通/超星（mooc2-ans.chaoxing.com、mooc1.chaoxing.com）课程里找出未完成的任务点，逐章完成「章节测验」并提交，最后汇报每章得分与错题。当用户给出学习通课程链接、要求“做没做完的任务点/章节测验”“测试做题正确度”“选择答案最后提交”时使用。
---

# 学习通章节测验自动化

## 何时使用

- 用户贴出学习通课程链接，说“有一些没有完成的任务点，进去做题并提交”。
- 用户要求测试答题正确率，或要求“把 X 章以后的题都做了”。
- 用户附加节奏/形式要求，例如“每个题的选择时间不超过 4s”“提交后还要再点一次提交”。

## 用户约束（硬性）

1. **每题选择用时 ≤ 4 秒**：选项要连续快速点完，不要在一道题上停留、反复截图。
2. **提交是两步**：先点页面上的「提交」，弹窗出现后再点弹窗里的「提交」（不是「确定」）。
3. **只做章节测验**，不刷视频、不写作业，除非用户明确要求。
4. **不要私自重做已提交的测验**：重做会产生“第 2 次作答”记录。只有用户要求冲满分时才重做。
5. 完成后汇报：每章得分、错题原因、平均分、剩余未完成任务点。

## 前置：登录

- 打开链接若被重定向到 `passport2.chaoxing.com/login`，说明自动化浏览器是独立配置，没有用户的登录态。
- 让用户用**学习通 App 扫码**登录（截图后告诉用户点右侧二维码），等用户回复“好了”再继续。
- 不要主动索要账号密码；用户主动给才用。

## 页面结构

- 顶层：`mooc1.chaoxing.com/mycourse/studentstudy?courseId=...&clazzid=...&cpi=...&enc=...&openc=...&chapterId=<id>`
- 子框架：
  - `iframe[name=iframe]` → `mooc-ans/knowledge/cards`（任务点标签：`1 视频` / `2 章节测验`）
  - `iframe[name=frame_content]` → `mooc-ans/work/doHomeWorkNew`（真正的题目页）
  - 提交后同一 frame 变成 `work/selectWorkQuestionYiPiYue`（成绩页）
- 切标签：在**顶层文档**里找 `span.spanText` 文本为「章节测验」的元素，点它的父 `li`；`li` 已带 `active` 就不用点。
- 题目容器：`.TiMu`
  - 单选 li：`onclick="addChoice(this)"`，选项字母在 `li span[data]`（A/B/C/D）
  - 多选 li：`onclick="addMultipleChoice(this)"`，字母在 `li span.num_option_dx[data]`
  - 判断题：选项 data 是 `true` / `false`（对 / 错）
  - 隐藏答案字段：`input[id^=answer<qid>]`，qid 在 `li[qid]`

## 读题：必须靠截图

- 题干带 `font-cxsecret` 字体加密，`innerText` 读出来是乱码，**只能用 `browser.screenshot(fullPage)` 看图读题**。
- 参数建议：`maxWidth: 1200~1300`、`format: "jpeg"`、`quality: 80`，清晰且省上下文。
- 截图报 `Screenshot needs a visible tab`：自动化浏览器窗口被最小化/被挡住，让用户还原窗口，别硬试。

## 逐章流程

1. 拿章节 ID：侧边栏元素 `onclick="getTeacherAjax(courseId, clazzId, chapterId)"`，取第三个参数；也可从大纲页 `input[type=checkbox]` 的 value 取。
2. `browser.navigate` 到 `studentstudy?...&chapterId=<id>`。
3. 轮询：点「章节测验」标签 → 取 `frame[name=frame_content]` → 直到 `document.querySelectorAll('.TiMu').length > 0`。
4. `browser.tabs.focus` 后 `browser.screenshot(fullPage)` 读题。
5. 一次 `evaluate` 内完成：按答案点选项 → `btnBlueSubmit()` → 轮询 `top.document.getElementById('popok')` 可见后点击。
6. 读成绩页文本，抓 `本次成绩(\d+(\.\d+)?)分`。

## 可复用代码片段

点选项 + 提交 + 二次确认（在 `frame_content` 里执行）：

```js
new Promise((resolve) => {
  const answers = [["A"], ["B"], ["true"], ["A","C"]]; // 按题号顺序
  [...document.querySelectorAll('.TiMu')].forEach((q, i) => {
    const want = answers[i] || [];
    [...q.querySelectorAll('.Zy_ulTop li')].forEach((li, j) => {
      const d = li.querySelector('span[data]')?.getAttribute('data');
      const pick = (d === 'true' || d === 'false')
        ? want.includes(d)
        : want.includes(String.fromCharCode(65 + j));
      if (pick) li.click();
    });
  });
  btnBlueSubmit();
  let n = 0;
  const iv = setInterval(() => {
    let b = null;
    try { b = top.document.getElementById('popok'); } catch (e) {}
    if (b && b.getBoundingClientRect().width > 0) {
      b.click(); clearInterval(iv); resolve('clicked'); return;
    }
    if (++n > 80) { clearInterval(iv); resolve('no modal'); }
  }, 100);
})
```

> 提交后页面会跳转，调用会抛 `context destroyed` / `Execution context was destroyed`，属正常，接着读新页面即可。

填空题（UEditor）：用已存在的实例，别用 `UE.getEditor()`（会重建编辑器并报 `Cannot read properties of null`）：

```js
UE.instants['ueditorInstant0'].setContent('价值增值');
```

## 题型与评分经验

- 多选题：全对满分；**少选通常给一半分**；选中错误选项给 0。
- 判断题常设陷阱：人名与朝代/著作张冠李戴（如《梦溪笔谈》不是宋应星、《出师表》不是曹植）、主体互换（“治人/治于人”“三孔/五孔插座”）、“只/完全/都”这类绝对化表述。
- 网络答案库（超星尔雅答案站）选项字母常与当前试卷顺序不一致，**只能按选项内容对照**，不能照抄字母；答案库本身也常有错。

## 收尾检查

- 回侧边栏看每个目录后的橙色数字：`2` = 视频+测验都没做，`1` = 只剩视频，绿勾 = 全完成。
- 用 `已没有任何目录显示 2` 来确认所有测验都已提交。
- 汇报表格：章节 / 得分 / 错题原因；再给平均分和剩余任务点（通常是视频）。

## 已知坑

- **离开或刷新未提交的测验页会自动提交当前已选答案**：没选完不要导航走。
- 有的章节首次打开只显示视频页，需要再点一次「章节测验」才加载题目。
- 提交确认弹窗在**顶层文档**（`top.document`），测验 iframe 里同名的「确认提交？」div 是隐藏模板，点了没反应。
- 确认按钮 id 是 `popok`，文案是「提交」；不是「确定」。
- `browser.wait` 的参数是 `condition: { text, exists }` 这类结构，不是 `text/timeoutMs`。
