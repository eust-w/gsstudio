import {describe, expect, it} from "vitest";
import {TaskDetail} from "../api";
import {viewerAccess} from "./ViewerPage";
const task: TaskDetail = {id:"scene",name:"Scene",type:"worldrecon",status:"completed",progress:1,created_at:"2026-10-03",config:{},stages:[],artifacts:[{name:"gaussians.spz",path:"output/gaussians.spz",type:"3dgs",size:100}]};
describe("standalone viewer access", () => {
  it("never lets direct links bypass a failed current quality gate", () => {
    expect(viewerAccess(task, {available:true,contract_version:4,free_view_eligible:false}, task.artifacts[0].path).allowed).toBe(false);
  });
  it("does not load arbitrary paths or demo models", () => {
    const quality = {available:true,contract_version:4,free_view_eligible:true};
    expect(viewerAccess(task, quality, "input/source.png").allowed).toBe(false);
    expect(viewerAccess({...task,execution_mode:"demo"}, quality, task.artifacts[0].path).allowed).toBe(false);
  });
  it("distinguishes legacy inspection from validated navigation", () => {
    expect(viewerAccess(task, {available:false}, task.artifacts[0].path)).toMatchObject({allowed:true,navigation:false});
    expect(viewerAccess(task, {available:true,contract_version:4,free_view_eligible:true}, task.artifacts[0].path)).toMatchObject({allowed:true,navigation:true});
  });
});
