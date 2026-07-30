const form = document.querySelector("#weather-form");
const dateInput = document.querySelector("#date");

if (form && dateInput) {
    form.addEventListener("submit", (event) => {
        if (!dateInput.value) {
            event.preventDefault();
            dateInput.focus();
            return;
        }

        const button = form.querySelector("button[type='submit']");
        if (button) {
            button.disabled = true;
            button.querySelector("span").textContent = "Predicting…";
        }
    });
}
