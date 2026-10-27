export interface Workspace {
  id: string;
  name: string;
  description: string;
  license: string;
  examples: string[];
}

export interface Me {
  id: string;
  name: string;
  groups: string[];
  workspaces: Workspace[];
}

export interface DemoUser {
  id: string;
  name: string;
  api_key: string;
  groups: string[];
  workspaces: string[];
}

export interface Source {
  n: number;
  chunk_id: string;
  title: string;
  headings: string[];
  url: string;
  text: string;
  score: number;
  lexical_rank: number | null;
  dense_rank: number | null;
  rerank_score: number | null;
  flags: string[];
}

export interface Answer {
  id: string;
  workspace: string;
  question: string;
  standalone_question: string;
  text: string;
  abstained: boolean;
  abstain_reason: string | null;
  cited: number[];
  invalid_citations: number[];
  grounded: boolean;
  sources: Source[];
  mode: string;
  timings_ms: Record<string, number>;
  usage: Record<string, number>;
}

export interface Turn {
  id: string;
  question: string;
  status: "searching" | "writing" | "done" | "error";
  draft: string;
  sources: Source[];
  answer?: Answer;
  error?: string;
  feedback?: 1 | -1;
}
