import type { AnswerSection as AnswerSectionType } from "../../api/types";

type Props = {
  section: {
    type: AnswerSectionType["type"];
    content: string;
    streaming?: boolean;
  };
};

function AnswerSection({ section }: Props) {
  return (
    <div style={{ marginTop: "0.75rem" }}>
      <div
        style={{
          fontSize: "0.75rem",
          fontWeight: 600,
          color: "#666",
          letterSpacing: "0.05em",
        }}
      >
        {section.type.toUpperCase()}
      </div>

      <p style={{ marginTop: "0.25rem", lineHeight: 1.6 }}>
        {section.content}
        {section.streaming && (
          <span style={{ opacity: 0.5, marginLeft: 2 }}>▍</span>
        )}
      </p>
    </div>
  );
}

export default AnswerSection;
