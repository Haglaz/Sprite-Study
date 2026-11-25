import { useState } from 'react';
import type { EvolutionGenerationPoint } from '../evolution/types';

interface TrainingCurvesChartProps {
  history: EvolutionGenerationPoint[];
  valCorruptedMedian: number;
  realCeilingMedian: number;
  junkFloorMedian: number;
}

export function TrainingCurvesChart({
  history,
  valCorruptedMedian,
  realCeilingMedian,
  junkFloorMedian,
}: TrainingCurvesChartProps) {
  const [selectedGen, setSelectedGen] = useState<number | null>(null);

  const width = 760;
  const height = 268;
  const padL = 48;
  const padR = 24;
  const padT = 18;
  const padB = 34;
  const plotW = width - padL - padR;
  const plotH = height - padT - padB;

  const yMin = 0;
  const yMax = 100;
  const toY = (v: number) => padT + plotH * (1 - Math.max(yMin, Math.min(yMax, v)) / (yMax - yMin));

  const maxGen = Math.max(1, history.length > 0 ? history[history.length - 1].generation : 1);
  const minGen = history.length > 0 ? history[0].generation : 1;
  const toX = (gen: number) => {
    if (history.length <= 1) return padL + plotW * 0.5;
    return padL + ((gen - minGen) / Math.max(1, maxGen - minGen)) * plotW;
  };

  const trainPath = history
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${toX(p.generation).toFixed(1)} ${toY(p.trainBestScore).toFixed(1)}`)
    .join(' ');
  const trainMeanPath = history
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${toX(p.generation).toFixed(1)} ${toY(p.trainMeanScore).toFixed(1)}`)
    .join(' ');
  const valPath = history
    .map((p, idx) => `${idx === 0 ? 'M' : 'L'} ${toX(p.generation).toFixed(1)} ${toY(p.valBestScore).toFixed(1)}`)
    .join(' ');

  const focusedPoint =
    history.find((p) => p.generation === selectedGen) ??
    (history.length > 0 ? history[history.length - 1] : null);

  const yTicks = [0, 25, 50, 75, 100];

  return (
    <div className="space-y-3" data-testid="train-val-curves-panel">
      {/* Legend & Focused Generation Readout */}
      <div className="flex flex-wrap items-center justify-between gap-3 text-xs">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[#A6B0C3]">
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block w-3 h-0.5 bg-[#E5A93C]" aria-hidden="true" />
            <span>Treino (melhor)</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block w-3 h-0.5 bg-[#60A5FA]" aria-hidden="true" />
            <span>Validação (média held-out)</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block w-3 border-t border-dashed border-[#22C55E]" aria-hidden="true" />
            <span>Teto real ({realCeilingMedian.toFixed(1)})</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block w-3 border-t border-dashed border-[#64748B]" aria-hidden="true" />
            <span>Corrompido ({valCorruptedMedian.toFixed(1)})</span>
          </span>
          <span className="inline-flex items-center gap-1.5">
            <span className="inline-block w-3 border-t border-dashed border-[#F87171]" aria-hidden="true" />
            <span>Piso lixo ({junkFloorMedian.toFixed(1)})</span>
          </span>
        </div>

        {focusedPoint && (
          <div className="font-mono tabular-nums text-[11px] text-[#CBD5E1] flex flex-wrap items-center gap-2">
            <span>Gen {focusedPoint.generation}</span>
            <span aria-hidden="true">·</span>
            <span className="text-[#E5A93C]">Treino {focusedPoint.trainBestScore.toFixed(1)}</span>
            <span aria-hidden="true">·</span>
            <span className="text-[#93C5FD]">Val {focusedPoint.valBestScore.toFixed(1)}</span>
            <span aria-hidden="true">·</span>
            <span className={focusedPoint.valDelta >= 0 ? 'text-[#86EFAC]' : 'text-[#FCA5A5]'}>
              {focusedPoint.valDelta >= 0 ? `+${focusedPoint.valDelta.toFixed(1)}` : focusedPoint.valDelta.toFixed(1)}
            </span>
            <span aria-hidden="true">·</span>
            <span>{focusedPoint.promoted ? 'Promovido' : 'Rejeitado'}</span>
          </div>
        )}
      </div>

      {history.length === 0 ? (
        <div className="h-[240px] rounded-lg border border-dashed border-[#262D3B] bg-[#0F1218] flex flex-col items-center justify-center text-center px-6 gap-1.5">
          <p className="text-sm font-medium text-[#CBD5E1]">
            Nenhuma geração executada nesta categoria
          </p>
          <p className="text-xs text-[#7D889D] max-w-md">
            Inicie o treino co-evolutivo para plotar as curvas reais de Treino vs Validação held-out ao lado do Teto Real ({realCeilingMedian.toFixed(1)}) e do Piso Lixo ({junkFloorMedian.toFixed(1)}).
          </p>
        </div>
      ) : (
        <div className="rounded-lg border border-[#202633] bg-[#0E1117] p-2 overflow-x-auto">
          <svg
            viewBox={`0 0 ${width} ${height}`}
            role="img"
            aria-label="Curvas de evolução Treino versus Validação held-out"
            className="w-full h-auto min-w-[540px] block"
          >
            {/* Horizontal grid lines & Y axis */}
            {yTicks.map((tick) => {
              const y = toY(tick);
              return (
                <g key={tick}>
                  <line
                    x1={padL}
                    y1={y}
                    x2={width - padR}
                    y2={y}
                    stroke="#1D2330"
                    strokeWidth={1}
                  />
                  <text
                    x={padL - 8}
                    y={y + 4}
                    textAnchor="end"
                    fill="#6E7A90"
                    fontSize={10}
                    fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
                  >
                    {tick}
                  </text>
                </g>
              );
            })}

            {/* Reference Baselines: Real Ceiling, Corrupted Validation Baseline, Junk Floor */}
            <line
              x1={padL}
              y1={toY(realCeilingMedian)}
              x2={width - padR}
              y2={toY(realCeilingMedian)}
              stroke="#22C55E"
              strokeWidth={1.25}
              strokeDasharray="5 4"
              opacity={0.75}
            />
            <line
              x1={padL}
              y1={toY(valCorruptedMedian)}
              x2={width - padR}
              y2={toY(valCorruptedMedian)}
              stroke="#64748B"
              strokeWidth={1.25}
              strokeDasharray="3 3"
              opacity={0.8}
            />
            <line
              x1={padL}
              y1={toY(junkFloorMedian)}
              x2={width - padR}
              y2={toY(junkFloorMedian)}
              stroke="#F87171"
              strokeWidth={1.25}
              strokeDasharray="5 4"
              opacity={0.75}
            />

            {/* Right-aligned baseline labels */}
            <text
              x={width - padR - 2}
              y={Math.max(padT + 10, toY(realCeilingMedian) - 4)}
              textAnchor="end"
              fill="#86EFAC"
              fontSize={9.5}
              fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
            >
              Teto Real {realCeilingMedian.toFixed(1)}
            </text>
            <text
              x={width - padR - 2}
              y={Math.min(height - padB - 4, toY(junkFloorMedian) + 11)}
              textAnchor="end"
              fill="#FCA5A5"
              fontSize={9.5}
              fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
            >
              Piso Lixo {junkFloorMedian.toFixed(1)}
            </text>

            {/* Train mean (subtle secondary line) */}
            {history.length > 1 && (
              <path
                d={trainMeanPath}
                fill="none"
                stroke="#E5A93C"
                strokeWidth={1}
                strokeDasharray="2 3"
                opacity={0.42}
              />
            )}

            {/* Train best curve */}
            {history.length > 1 && (
              <path
                d={trainPath}
                fill="none"
                stroke="#E5A93C"
                strokeWidth={2}
                strokeLinejoin="round"
              />
            )}

            {/* Validation held-out median curve */}
            {history.length > 1 && (
              <path
                d={valPath}
                fill="none"
                stroke="#60A5FA"
                strokeWidth={2.25}
                strokeLinejoin="round"
              />
            )}

            {/* Interactive generation nodes + X axis labels */}
            {history.map((p) => {
              const x = toX(p.generation);
              const yTr = toY(p.trainBestScore);
              const yVal = toY(p.valBestScore);
              const isSelected = focusedPoint?.generation === p.generation;
              const showLabel =
                history.length <= 12 ||
                p.generation === 1 ||
                p.generation === maxGen ||
                p.generation % Math.ceil(history.length / 8) === 0;

              return (
                <g
                  key={p.generation}
                  onMouseEnter={() => setSelectedGen(p.generation)}
                  onClick={() => setSelectedGen(p.generation)}
                  className="cursor-pointer"
                >
                  {isSelected && (
                    <line
                      x1={x}
                      y1={padT}
                      x2={x}
                      y2={height - padB}
                      stroke="#334155"
                      strokeWidth={1}
                    />
                  )}
                  <circle
                    cx={x}
                    cy={yTr}
                    r={isSelected ? 4 : 3}
                    fill="#E5A93C"
                  />
                  {p.promoted ? (
                    <polygon
                      points={`${x},${yVal - 5.5} ${x + 5.5},${yVal} ${x},${yVal + 5.5} ${x - 5.5},${yVal}`}
                      fill="#22C55E"
                      stroke="#0E1117"
                      strokeWidth={1.2}
                    />
                  ) : (
                    <circle
                      cx={x}
                      cy={yVal}
                      r={isSelected ? 4.5 : 3.5}
                      fill="#0E1117"
                      stroke="#60A5FA"
                      strokeWidth={1.8}
                    />
                  )}
                  {showLabel && (
                    <text
                      x={x}
                      y={height - 12}
                      textAnchor="middle"
                      fill={isSelected ? '#F1F5F9' : '#6E7A90'}
                      fontSize={10}
                      fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
                    >
                      G{p.generation}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        </div>
      )}
    </div>
  );
}
