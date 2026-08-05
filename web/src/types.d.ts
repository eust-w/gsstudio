declare module "@mkkellogg/gaussian-splats-3d" {
  export enum SceneFormat {
    Splat = 0,
    Ply = 1,
    Spz = 2,
  }

  export interface ViewerOptions {
    cameraUp?: number[];
    initialCameraPosition?: number[];
    initialCameraLookAt?: number[];
    rootElement?: HTMLElement;
    selfDrivenMode?: boolean;
    sharedMemoryForWorkers?: boolean;
  }

  export interface SplatSceneOptions {
    showLoadingUI?: boolean;
    format?: SceneFormat;
    onProgress?: (progress: number) => void;
  }

  export class Viewer {
    constructor(options?: ViewerOptions);
    addSplatScene(url: string, options?: SplatSceneOptions): Promise<void>;
    dispose(): void;
  }
}
