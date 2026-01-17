import type { MessageMetadata } from "../../api/types";
import AnswerSection from "./AnswerSection";

type Props = {
  sections: {
    type: string;
    content: string;
    streaming?: boolean;
  }[];
  metadata?: MessageMetadata;
};

function AnswerBlock({ sections }: Props) {
  return (
    <div style={{ marginBottom: "1.5rem" }}>
      <strong>AESOP</strong>

      {sections.map((section, idx) => (
        <AnswerSection key={idx} section={section} />
      ))}
    </div>
  );
}

export default AnswerBlock;
