import { useState } from "react";

type Props = {
  onSend: (text: string) => void;
};

function ChatInput({ onSend }: Props) {
  const [value, setValue] = useState("");

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!value.trim()) return;

    onSend(value);
    setValue("");
  };

  return (
    <form onSubmit={handleSubmit} style={{ marginTop: "1rem" }}>
      <textarea
        value={value}
        onChange={(e) => setValue(e.target.value)}
        rows={3}
        placeholder="Ask a clinical or research question..."
        style={{ width: "100%", resize: "none" }}
      />
      <button style={{ marginTop: "0.5rem" }}>Ask</button>
    </form>
  );
}

export default ChatInput;
