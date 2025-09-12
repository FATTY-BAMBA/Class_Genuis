document.addEventListener("DOMContentLoaded", function () {
    let checkButtons = document.querySelectorAll(".check-me");
    let submitButton = document.getElementById("submit-answers");
    let scoreDisplay = document.getElementById("score-display");

    // Disable check buttons initially
    checkButtons.forEach(button => button.disabled = true);

    // Event listener for submitting all answers
    submitButton.addEventListener("click", function () {
        let totalQuestions = document.querySelectorAll(".question").length;
        let score = 0;

        document.querySelectorAll(".question").forEach((questionDiv, index) => {
            let selected = document.querySelector(`input[name="q${index + 1}"]:checked`);
            let correctAnswer = questionDiv.getAttribute("data-answer").trim().toUpperCase();
            let feedbackDiv = questionDiv.querySelector(".feedback");

            if (selected) {
                let userAnswer = selected.value.trim().toUpperCase();
                if (userAnswer === correctAnswer) {
                    questionDiv.classList.add("correct");
                    feedbackDiv.innerHTML = `<span style="color:green;">✅ 正確！</span>`;
                    score++;
                } else {
                    questionDiv.classList.add("incorrect");
                    feedbackDiv.innerHTML = `<span style="color:red;">❌ 錯誤，正確答案是：${correctAnswer}</span>`;
                }
                checkButtons[index].disabled = false;
            }
        });

        scoreDisplay.innerHTML = `📊 分數：${score} / ${totalQuestions}`;
    });

    // Event listener for checking individual answers
    checkButtons.forEach((button, index) => {
        button.addEventListener("click", function () {
            let questionDiv = this.parentElement;
            let selected = questionDiv.querySelector("input:checked");
            let correctAnswer = questionDiv.getAttribute("data-answer").trim().toUpperCase();
            let explanation = questionDiv.getAttribute("data-explanation").trim();
            let feedbackDiv = questionDiv.querySelector(".feedback");

            if (!selected) {
                feedbackDiv.innerHTML = "⚠️ 請先選擇一個答案！";
                return;
            }

            let userAnswer = selected.value.trim().toUpperCase();

            if (userAnswer === correctAnswer) {
                questionDiv.classList.add("correct");
                feedbackDiv.innerHTML = `
                    <span style="color:green;">✅ 正確！</span><br>
                    <strong>解釋：</strong><br>
                    ${explanation.replace(/\n/g, "<br>")}
                `;
            } else {
                questionDiv.classList.add("incorrect");
                feedbackDiv.innerHTML = `
                    <span style="color:red;">❌ 錯誤，正確答案是：${correctAnswer}</span><br>
                    <strong>解釋：</strong><br>
                    ${explanation.replace(/\n/g, "<br>")}
                `;
            }

            this.disabled = true;
        });
    });

    if (typeof Prism !== "undefined") {
        Prism.highlightAll();
    } else {
        console.warn("⚠️ Prism.js not loaded correctly.");
    }

    function highlightPrismCode() {
        if (typeof Prism !== "undefined") {
            Prism.highlightAll();
            console.log("🎨 Prism.js highlighting applied successfully.");
        } else {
            console.error("❌ Prism.js is not defined.");
        }
    }

    setTimeout(highlightPrismCode, 500);
    setTimeout(highlightPrismCode, 1000);
    setTimeout(highlightPrismCode, 1500);
});
