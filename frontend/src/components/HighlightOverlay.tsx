import React, { useState } from 'react';

export interface Keypoint {
  x: number;
  y: number;
  type?: string;
  label?: string;
}

export interface Region {
  points: [number, number][];
  type?: string;
  label?: string;
}

export interface HighlightOverlayProps {
  width: number; // 显示图片宽度
  height: number; // 显示图片高度
  imageWidth: number; // 原图宽度
  imageHeight: number; // 原图高度
  keypoints?: Keypoint[];
  regions?: Region[];
}

const colorMap: Record<string, string> = {
  eye: 'rgba(0,153,255,',
  nose: 'rgba(255,102,0,',
  mouth: 'rgba(255,0,128,',
  default: 'rgba(255,0,0,',
};

function scalePoint(
  x: number,
  y: number,
  originalWidth: number,
  originalHeight: number,
  displayWidth: number,
  displayHeight: number
) {
  return {
    x: (x / originalWidth) * displayWidth,
    y: (y / originalHeight) * displayHeight,
  };
}

const HighlightOverlay: React.FC<HighlightOverlayProps> = ({
  width,
  height,
  imageWidth,
  imageHeight,
  keypoints,
  regions,
}) => {
  const [hovered, setHovered] = useState<string | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  return (
    <svg
      width={width}
      height={height}
      style={{
        position: 'absolute',
        left: 0,
        top: 0,
        pointerEvents: 'none', // 不影响鼠标操作
        zIndex: 2,
      }}
    >
      {/* 高亮关键点 */}
      {keypoints?.map((kp, idx) => {
        const { x, y } = scalePoint(kp.x, kp.y, imageWidth, imageHeight, width, height);
        const id = `kp-${idx}`;
        const color = colorMap[kp.type || 'default'] || colorMap.default;
        return (
          <g key={id}>
            <circle
              cx={x}
              cy={y}
              r={hovered === id || selected === id ? 18 : 14}
              fill={color + (hovered === id || selected === id ? '0.7)' : '0.4)')}
              stroke={selected === id ? '#ff0' : '#fff'}
              strokeWidth={selected === id ? 4 : 2}
              style={{ pointerEvents: 'auto', cursor: 'pointer' }}
              onMouseEnter={() => setHovered(id)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => setSelected(id)}
            >
              <title>{kp.label || kp.type}</title>
            </circle>
            {/* 可选：显示标签 */}
            {(hovered === id || selected === id) && (
              <text x={x + 20} y={y - 10} fontSize={16} fill="#333" stroke="#fff" strokeWidth={0.5}>
                {kp.label || kp.type}
              </text>
            )}
          </g>
        );
      })}
      {/* 高亮区域 */}
      {regions?.map((region, idx) => {
        const id = `region-${idx}`;
        const color = colorMap[region.type || 'default'] || colorMap.default;
        const points = region.points
          .map(([x, y]) => {
            const { x: sx, y: sy } = scalePoint(x, y, imageWidth, imageHeight, width, height);
            return `${sx},${sy}`;
          })
          .join(' ');
        // 计算区域中心点用于标签
        const center = region.points.reduce(
          (acc, [x, y]) => [acc[0] + x, acc[1] + y],
          [0, 0]
        ).map(v => v / region.points.length);
        const { x: cx, y: cy } = scalePoint(center[0], center[1], imageWidth, imageHeight, width, height);
        return (
          <g key={id}>
            <polygon
              points={points}
              fill={color + (hovered === id || selected === id ? '0.5)' : '0.25)')}
              stroke={selected === id ? '#ff0' : '#fff'}
              strokeWidth={selected === id ? 4 : 2}
              style={{ pointerEvents: 'auto', cursor: 'pointer' }}
              onMouseEnter={() => setHovered(id)}
              onMouseLeave={() => setHovered(null)}
              onClick={() => setSelected(id)}
            >
              <title>{region.label || region.type}</title>
            </polygon>
            {(hovered === id || selected === id) && (
              <text x={cx + 20} y={cy - 10} fontSize={16} fill="#333" stroke="#fff" strokeWidth={0.5}>
                {region.label || region.type}
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
};

export default HighlightOverlay; 